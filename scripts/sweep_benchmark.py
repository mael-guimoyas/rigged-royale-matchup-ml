"""Time a full meta sweep against a live inference service, over HTTP.

Single-batch curl timings do not answer the question that matters. A real sweep
sends tens of batches concurrently over pooled connections, so per-request TLS
setup is paid once per connection rather than once per batch -- which is exactly
what a loop of separate curl processes gets wrong, and why it understates a
remote endpoint.

This mirrors what the site does (see riggedroyale ml-inference.ts): batches of
ML_INFERENCE_BATCH_SIZE rows, ML_INFERENCE_BATCH_CONCURRENCY of them in flight,
one persistent connection per worker.

Run it from the machine that will really be calling the service -- netcup, not a
laptop -- or the numbers describe your home uplink instead of production.

    python3 scripts/sweep_benchmark.py --url http://localhost:8080 \
        --api-key "$PK" --rows 100000 --batch 512 --concurrency 3

    python3 scripts/sweep_benchmark.py --url https://<id>.api.runpod.ai \
        --bearer "$KEY" --rows 100000 --batch 2048 --concurrency 3

Only stdlib, so it runs on a bare server with no pip install.
"""

from __future__ import annotations

import argparse
import http.client
import json
import math
import random
import ssl
import statistics
import threading
import time
from pathlib import Path
from queue import Empty, Queue
from urllib.parse import urlparse

DEFAULT_EXAMPLE = Path(__file__).resolve().parents[1] / "example-matchup.json"


def _variants(example: dict, count: int, seed: int) -> list[dict]:
    """Distinct requests built by reshuffling the example's own card ids.

    Reusing ids that are already in the checkpoint's vocabulary keeps every row
    valid without needing the card table here. Distinct decks matter: a sweep
    never sends the same matchup thousands of times, and identical rows would
    flatter the encoder's branch prediction.
    """
    pool = list(example["team_card_ids"]) + list(example["opponent_card_ids"])
    rng = random.Random(seed)
    out = []
    for _ in range(count):
        shuffled = pool[:]
        rng.shuffle(shuffled)
        row = dict(example)
        row["team_card_ids"] = shuffled[:8]
        row["opponent_card_ids"] = shuffled[8:16]
        # Evolution/hero ids must stay a subset of the deck they belong to.
        row["team_evolution_card_ids"] = shuffled[:1]
        row["opponent_evolution_card_ids"] = shuffled[8:9]
        row["team_hero_card_ids"] = []
        row["opponent_hero_card_ids"] = []
        out.append(row)
    return out


class _Client:
    """One persistent HTTP(S) connection, reconnecting if the peer drops it."""

    def __init__(self, url: str, headers: dict[str, str], timeout: float) -> None:
        parsed = urlparse(url)
        self.host = parsed.hostname or ""
        self.secure = parsed.scheme == "https"
        self.port = parsed.port or (443 if self.secure else 80)
        self.path = (parsed.path or "").rstrip("/") + "/predict/batch"
        self.headers = headers
        self.timeout = timeout
        self.connection: http.client.HTTPConnection | None = None

    def _connect(self) -> http.client.HTTPConnection:
        if self.secure:
            return http.client.HTTPSConnection(
                self.host, self.port, timeout=self.timeout, context=ssl.create_default_context()
            )
        return http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)

    def post(self, body: bytes) -> tuple[int, int]:
        """Send one batch; returns (status, prediction count). Retries once on a
        dropped keep-alive, which is a normal event, not a service failure."""
        for attempt in (0, 1):
            if self.connection is None:
                self.connection = self._connect()
            try:
                self.connection.request("POST", self.path, body=body, headers=self.headers)
                response = self.connection.getresponse()
                payload = response.read()
                if response.status != 200:
                    return response.status, 0
                return 200, len(json.loads(payload).get("predictions", []))
            except (http.client.HTTPException, OSError):
                try:
                    if self.connection:
                        self.connection.close()
                finally:
                    self.connection = None
                if attempt == 1:
                    return 0, 0
        return 0, 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="Base URL of the inference service.")
    parser.add_argument("--bearer", default=None, help="Authorization: Bearer token (RunPod).")
    parser.add_argument("--api-key", default=None, help="X-API-Key header (PREDICT_API_KEY).")
    parser.add_argument("--rows", type=int, default=100_000, help="Total matchups to sweep.")
    parser.add_argument("--batch", type=int, default=2048, help="Rows per request.")
    parser.add_argument("--concurrency", type=int, default=3, help="Requests in flight.")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--example", type=Path, default=DEFAULT_EXAMPLE)
    parser.add_argument("--seed", type=int, default=17)
    arguments = parser.parse_args()

    example = json.loads(arguments.example.read_text(encoding="utf-8"))
    headers = {"Content-Type": "application/json"}
    if arguments.bearer:
        headers["Authorization"] = f"Bearer {arguments.bearer}"
    if arguments.api_key:
        headers["X-API-Key"] = arguments.api_key

    batches = [
        min(arguments.batch, arguments.rows - start)
        for start in range(0, arguments.rows, arguments.batch)
    ]
    print(f"target   {arguments.url}")
    print(f"sweep    {arguments.rows} rows, {len(batches)} batches of {arguments.batch}, "
          f"{arguments.concurrency} in flight")

    # Build the payloads up front so serialisation is not timed as service cost.
    print("building payloads...", flush=True)
    bodies = [
        json.dumps({"requests": _variants(example, size, arguments.seed + index)}).encode()
        for index, size in enumerate(batches)
    ]
    megabytes = sum(len(body) for body in bodies) / (1024 * 1024)
    print(f"payload  {megabytes:.1f} MiB total")

    # Warm every worker the sweep will use, not just one. A remote endpoint
    # scales workers on concurrent request count, so a single warm-up batch
    # leaves the other --concurrency-1 workers cold and they pay their start-up
    # inside the measurement. Firing `concurrency` batches at once forces the
    # endpoint to spin them all up first.
    print(f"warming up {arguments.concurrency} worker(s)...", flush=True)
    warm_results: list[int] = []
    warm_lock = threading.Lock()

    def warm_one() -> None:
        status, _ = _Client(arguments.url, headers, arguments.timeout).post(bodies[0])
        with warm_lock:
            warm_results.append(status)

    warmers = [threading.Thread(target=warm_one) for _ in range(max(1, arguments.concurrency))]
    for thread in warmers:
        thread.start()
    for thread in warmers:
        thread.join()
    if any(status != 200 for status in warm_results):
        print(f"WARNING: warm-up statuses {warm_results}")

    queue: Queue = Queue()
    for item in enumerate(bodies):
        queue.put(item)
    durations: list[float] = []
    answered = 0
    failures: dict[int, int] = {}
    lock = threading.Lock()

    def worker() -> None:
        nonlocal answered
        client = _Client(arguments.url, headers, arguments.timeout)
        while True:
            try:
                _, body = queue.get_nowait()
            except Empty:
                return
            mark = time.perf_counter()
            status, count = client.post(body)
            elapsed = time.perf_counter() - mark
            with lock:
                durations.append(elapsed)
                if status == 200:
                    answered += count
                else:
                    failures[status] = failures.get(status, 0) + 1
            queue.task_done()

    print("sweeping...", flush=True)
    started = time.perf_counter()
    threads = [threading.Thread(target=worker) for _ in range(max(1, arguments.concurrency))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    wall = time.perf_counter() - started

    ordered = sorted(durations)
    print()
    print(f"  answered      {answered} / {arguments.rows} rows")
    if failures:
        print(f"  FAILURES      {failures}  (status 0 = connection error)")
    print(f"  wall clock    {wall:.2f} s")
    print(f"  throughput    {answered / wall:,.0f} rows/s")
    if ordered:
        # Nearest-rank percentile: int(0.95 * (n - 1)) collapses below the median
        # on a handful of samples, which makes a smoke run look broken.
        def rank(fraction: float) -> float:
            return ordered[min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)]

        print(f"  per batch     min {ordered[0]:.3f}   p50 {statistics.median(ordered):.3f}   "
              f"p90 {rank(0.90):.3f}   p95 {rank(0.95):.3f}   max {ordered[-1]:.3f}  (s)")
        # Throughput the endpoint sustains once nothing is starting up. Batches
        # slower than twice the median are almost always a worker spinning up,
        # and reporting both numbers separates "this endpoint is slow" from
        # "this endpoint spent the sweep scaling".
        median = statistics.median(ordered)
        steady = [value for value in ordered if value <= 2 * median]
        if steady and len(steady) < len(ordered):
            per_batch = statistics.mean(steady)
            print(f"  steady state  {len(steady)}/{len(ordered)} batches under 2x median; "
                  f"implies {arguments.batch * arguments.concurrency / per_batch:,.0f} rows/s")
            print(f"  cold-start cost {(1 - (answered / wall) / (arguments.batch * arguments.concurrency / per_batch)) * 100:.0f}% "
                  "of the sweep's throughput")


if __name__ == "__main__":
    main()
