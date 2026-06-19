"""Collect battles straight from the Clash Royale API into training Parquet.

This bypasses the Postgres `battles.raw` archive entirely: it fetches each
player's battlelog (and profile, for the Path-of-Legends league), parses every
1v1 ranked/ladder battle with the same `domain.parse_battle_row` used by the
Supabase extractor, and writes training-ready Parquet shards locally and/or to
Supabase Storage. Opponents discovered in each battlelog are snowballed back
into the queue, so a small seed of tags expands into broad coverage.

Storage is object storage, not SQL: the shards here are consumed in bulk by
`prepare`/training, not queried per row. That is exactly what the model needs.
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq
from dotenv import load_dotenv
from tqdm import tqdm

from .config import AppConfig
from .domain import parse_battle_row
from .extraction import SCHEMA, Deduplicator

DEFAULT_BASE_URL = "https://proxy.royaleapi.dev/v1"
# The RoyaleAPI proxy sits behind Cloudflare, which rejects the default
# "Python-urllib/x" User-Agent with error 1010. A normal UA is required.
USER_AGENT = "rigged-royale-matchup-ml/0.1 (+https://github.com/riggedroyale)"
RANKED_BATTLE_TYPES = {"pathoflegend", "pathoflegends"}
LADDER_BATTLE_TYPES = {"pvp"}


class RateLimiter:
    """Thread-safe global request pacing shared across collector workers."""

    def __init__(self, requests_per_second: float) -> None:
        self._min_interval = 1.0 / max(requests_per_second, 0.1)
        self._lock = threading.Lock()
        self._next_slot = 0.0

    def acquire(self) -> None:
        with self._lock:
            now = time.perf_counter()
            slot = max(now, self._next_slot)
            self._next_slot = slot + self._min_interval
        delay = slot - time.perf_counter()
        if delay > 0:
            time.sleep(delay)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_tag(tag: str) -> str:
    """Uppercase, strip, ensure a single leading '#'. Mirrors the app."""
    cleaned = str(tag).strip().upper().replace("O", "0")
    cleaned = cleaned.lstrip("#")
    return f"#{cleaned}" if cleaned else ""


def _encode_tag(tag: str) -> str:
    return urllib.parse.quote(normalize_tag(tag), safe="")


def mode_key_for(battle: dict[str, Any]) -> str:
    """Replicate the ranked/ladder subset of the app's `battleModeFor`."""
    battle_type = str(battle.get("type") or "").lower()
    mode_name = str((battle.get("gameMode") or {}).get("name") or "").lower()
    if battle_type in RANKED_BATTLE_TYPES or mode_name.startswith("ranked"):
        return "ranked"
    if (
        battle_type in LADDER_BATTLE_TYPES
        or mode_name.startswith("ladder")
        or mode_name.startswith("seasonal")
    ):
        return "ladder"
    return "other"


def league_from_profile(profile: dict[str, Any] | None) -> int | None:
    if not profile:
        return None
    result = profile.get("currentPathOfLegendSeasonResult")
    if isinstance(result, dict):
        league = result.get("leagueNumber")
        if isinstance(league, int) and league > 0:
            return league
    return None


def _battle_fingerprint(player_tag: str, battle: dict[str, Any]) -> str:
    opponent = (battle.get("opponent") or [{}])[0]
    payload = json.dumps(
        [
            normalize_tag(player_tag),
            battle.get("battleTime"),
            opponent.get("tag"),
            [card.get("id") for card in opponent.get("cards", [])],
        ],
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def _opponent_tags(battle: dict[str, Any]) -> list[str]:
    tags: list[str] = []
    for participant in (battle.get("opponent") or []) + (battle.get("team") or []):
        tag = participant.get("tag")
        if tag:
            tags.append(normalize_tag(tag))
    return tags


class ClashRoyaleClient:
    def __init__(self, limiter: RateLimiter, max_retries: int = 4) -> None:
        token = (os.getenv("CR_API_TOKEN") or "").strip().strip("\"'").replace(" ", "")
        if not token:
            raise RuntimeError("CR_API_TOKEN is missing in .env")
        self._token = token
        self._base_url = (os.getenv("CR_API_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self._limiter = limiter
        self._max_retries = max_retries
        self._stats_lock = threading.Lock()
        self.requests = 0
        self.rate_limited = 0
        self.errors = 0

    def _bump(self, field: str) -> None:
        with self._stats_lock:
            setattr(self, field, getattr(self, field) + 1)

    def _get(self, path: str) -> Any | None:
        request = urllib.request.Request(
            f"{self._base_url}{path}",
            headers={
                "Authorization": f"Bearer {self._token}",
                "Accept": "application/json",
                "User-Agent": USER_AGENT,
            },
        )
        for attempt in range(self._max_retries):
            self._limiter.acquire()
            self._bump("requests")
            try:
                with urllib.request.urlopen(request, timeout=20) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as error:
                if error.code == 404:
                    return None
                if error.code == 429:
                    self._bump("rate_limited")
                if error.code in (429, 500, 502, 503, 504) and attempt < self._max_retries - 1:
                    time.sleep(2.0 * (attempt + 1))
                    continue
                self._bump("errors")
                raise
            except (urllib.error.URLError, TimeoutError):
                if attempt < self._max_retries - 1:
                    time.sleep(2.0 * (attempt + 1))
                    continue
                self._bump("errors")
                raise
        return None

    def player(self, tag: str) -> dict[str, Any] | None:
        return self._get(f"/players/{_encode_tag(tag)}")

    def battlelog(self, tag: str) -> list[dict[str, Any]]:
        result = self._get(f"/players/{_encode_tag(tag)}/battlelog")
        return result if isinstance(result, list) else []


class StorageClient:
    """Minimal Supabase Storage REST client (bucket + list/upload/download)."""

    def __init__(self, bucket: str, prefix: str, create: bool = False) -> None:
        url = (os.getenv("SUPABASE_URL") or "").rstrip("/")
        key = os.getenv("SUPABASE_SECRET_KEY") or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        if not url or not key:
            raise RuntimeError(
                "SUPABASE_URL and SUPABASE_SECRET_KEY (or SUPABASE_SERVICE_ROLE_KEY) "
                "are required for Supabase Storage."
            )
        self._url = url
        self._key = key
        self._bucket = bucket
        self._prefix = prefix.strip("/")
        if create:
            self._ensure_bucket()

    def _headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        headers = {"Authorization": f"Bearer {self._key}", "apikey": self._key}
        headers.update(extra or {})
        return headers

    def _object_path(self, object_name: str) -> str:
        return f"{self._prefix}/{object_name}" if self._prefix else object_name

    def _ensure_bucket(self) -> None:
        body = json.dumps({"id": self._bucket, "name": self._bucket, "public": False})
        request = urllib.request.Request(
            f"{self._url}/storage/v1/bucket",
            data=body.encode("utf-8"),
            headers=self._headers({"Content-Type": "application/json"}),
            method="POST",
        )
        try:
            urllib.request.urlopen(request, timeout=30).read()
        except urllib.error.HTTPError as error:
            if error.code not in (400, 409):  # already exists
                raise

    def upload(self, object_name: str, data: bytes) -> None:
        request = urllib.request.Request(
            f"{self._url}/storage/v1/object/{self._bucket}/"
            f"{urllib.parse.quote(self._object_path(object_name))}",
            data=data,
            headers=self._headers(
                {"Content-Type": "application/octet-stream", "x-upsert": "true"}
            ),
            method="POST",
        )
        urllib.request.urlopen(request, timeout=120).read()

    def list_objects(self, page: int = 1000) -> list[str]:
        names: list[str] = []
        offset = 0
        while True:
            body = json.dumps(
                {
                    "prefix": f"{self._prefix}/" if self._prefix else "",
                    "limit": page,
                    "offset": offset,
                    "sortBy": {"column": "name", "order": "asc"},
                }
            )
            request = urllib.request.Request(
                f"{self._url}/storage/v1/object/list/{self._bucket}",
                data=body.encode("utf-8"),
                headers=self._headers({"Content-Type": "application/json"}),
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                items = json.loads(response.read().decode("utf-8"))
            if not items:
                break
            names.extend(item["name"] for item in items if item.get("name"))
            if len(items) < page:
                break
            offset += page
        return names

    def download(self, object_name: str) -> bytes:
        request = urllib.request.Request(
            f"{self._url}/storage/v1/object/{self._bucket}/"
            f"{urllib.parse.quote(self._object_path(object_name))}",
            headers=self._headers(),
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.read()


def download_from_storage(
    config: AppConfig,
    bucket: str = "training-battles",
    prefix: str = "battles",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Download training Parquet shards from Supabase Storage into data/raw.

    The storage counterpart of `extract`: instead of paginating Postgres, it
    pulls the shards previously written by `collect-api --upload` so `prepare`
    and training can run without the database.
    """
    load_dotenv(config.resolve(".env"))
    raw_dir = config.resolve(config.data["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    client = StorageClient(bucket, prefix, create=False)
    names = [name for name in client.list_objects() if name.endswith(".parquet")]
    summary = {
        "objects_found": len(names),
        "downloaded": 0,
        "skipped_existing": 0,
        "raw_dir": str(raw_dir),
        "bucket": bucket,
    }
    for name in tqdm(names, desc="Storage download", unit="file"):
        destination = raw_dir / Path(name).name
        if destination.exists() and not overwrite:
            summary["skipped_existing"] += 1
            continue
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_bytes(client.download(name))
        temporary.replace(destination)
        summary["downloaded"] += 1
    return summary


def _seed_tags(
    config: AppConfig,
    tags_file: Path | None,
    from_supabase: int,
) -> list[str]:
    seeds: list[str] = []
    if tags_file is not None:
        for line in tags_file.read_text(encoding="utf-8").splitlines():
            tag = normalize_tag(line)
            if tag:
                seeds.append(tag)
    if from_supabase > 0:
        seeds.extend(_supabase_tags(from_supabase))
    return list(dict.fromkeys(seeds))


def _supabase_tags(limit: int) -> list[str]:
    import psycopg
    from psycopg.rows import tuple_row

    database_url = os.getenv("SUPABASE_DB_URL")
    if not database_url:
        raise RuntimeError("SUPABASE_DB_URL is missing; cannot seed tags from Supabase.")
    with psycopg.connect(database_url, row_factory=tuple_row) as connection:
        connection.execute("set default_transaction_read_only = on")
        rows = connection.execute(
            "select tag from public.players order by last_analyzed_at desc nulls last limit %s",
            (limit,),
        ).fetchall()
    return [normalize_tag(row[0]) for row in rows if row[0]]


def _next_shard_index(raw_dir: Path) -> int:
    indices = [
        int(path.stem.split("-")[-1])
        for path in raw_dir.glob("api-part-*.parquet")
        if path.stem.split("-")[-1].isdigit()
    ]
    return max(indices, default=-1) + 1


def _fetch_player(
    client: ClashRoyaleClient,
    tag: str,
    data_config: dict[str, Any],
    allowed_modes: set[str],
    snowball: bool,
) -> tuple[list[dict[str, Any]], list[str], int]:
    """Network worker: battlelog first, profile only if a ranked battle exists.

    Skipping the profile fetch for ladder-only players roughly halves the request
    count. Returns parsed records, opponent tags to enqueue, and battles seen.
    No dedup or file writes happen here so it is safe to run on many threads.
    """
    try:
        battles = client.battlelog(tag)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
        return [], [], 0
    league: int | None = None
    if any(mode_key_for(battle) == "ranked" for battle in battles):
        try:
            league = league_from_profile(client.player(tag))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
            league = None
    records: list[dict[str, Any]] = []
    opponents: list[str] = []
    for battle in battles:
        if snowball:
            opponents.extend(_opponent_tags(battle))
        mode_key = mode_key_for(battle)
        if allowed_modes and mode_key not in allowed_modes:
            continue
        row = {
            "raw": battle,
            "fingerprint": _battle_fingerprint(tag, battle),
            "battle_time": battle.get("battleTime"),
            "inserted_at": _now_iso(),
            "mode_key": mode_key,
            "league_number": league if mode_key == "ranked" else None,
        }
        parsed = parse_battle_row(row, data_config)
        if parsed is not None:
            records.append(parsed)
    return records, opponents, len(battles)


def collect_from_api(
    config: AppConfig,
    tags_file: Path | None = None,
    from_supabase: int = 0,
    snowball: bool = True,
    max_battles: int | None = None,
    max_players: int | None = None,
    requests_per_second: float = 30.0,
    workers: int = 16,
    upload: bool = False,
    bucket: str = "training-battles",
    prefix: str = "battles",
    shard_size: int = 50_000,
) -> dict[str, Any]:
    load_dotenv(config.resolve(".env"))
    seeds = _seed_tags(config, tags_file, from_supabase)
    if not seeds:
        raise RuntimeError("No seed tags. Pass --tags-file or --from-supabase.")

    raw_dir = config.resolve(config.data["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    limiter = RateLimiter(requests_per_second)
    client = ClashRoyaleClient(limiter)
    uploader = StorageClient(bucket, prefix, create=True) if upload else None
    deduplicator = Deduplicator(config.resolve(config.data["dedup_db"]))
    allowed_modes = set(config.data.get("allowed_modes") or [])

    queue: deque[str] = deque(seeds)
    queued: set[str] = set(seeds)
    shard_index = _next_shard_index(raw_dir)
    buffer: list[dict[str, Any]] = []
    summary = {
        "players_processed": 0,
        "battles_seen": 0,
        "battles_accepted": 0,
        "shards_written": 0,
        "uploaded": 0,
        "seed_tags": len(seeds),
    }
    progress = tqdm(desc="CR API players", unit="player")

    def flush() -> None:
        nonlocal shard_index
        if not buffer:
            return
        table = pa.Table.from_pylist(buffer, schema=SCHEMA)
        name = f"api-part-{shard_index:06d}.parquet"
        path = raw_dir / name
        pq.write_table(table, path, compression="zstd", row_group_size=50_000)
        if uploader is not None:
            uploader.upload(name, path.read_bytes())
            summary["uploaded"] += 1
        summary["shards_written"] += 1
        shard_index += 1
        buffer.clear()

    # Continuous dispatch: keep ~2x workers requests in flight and refill as each
    # finishes, so a slow or retrying player never stalls the others (a fixed-size
    # batch barrier wastes the pool on the slowest player in every round).
    target_inflight = max(workers * 2, workers + 4)
    submitted = 0

    def can_submit() -> bool:
        if not queue:
            return False
        if max_players is not None and submitted >= max_players:
            return False
        if max_battles is not None and summary["battles_accepted"] >= max_battles:
            return False
        return True

    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            inflight: dict[Future, str] = {}

            def fill() -> None:
                nonlocal submitted
                while can_submit() and len(inflight) < target_inflight:
                    tag = queue.popleft()
                    future = executor.submit(
                        _fetch_player, client, tag, config.data, allowed_modes, snowball
                    )
                    inflight[future] = tag
                    submitted += 1

            fill()
            while inflight:
                done, _ = wait(list(inflight), return_when=FIRST_COMPLETED)
                for future in done:
                    inflight.pop(future)
                    records, opponents, seen = future.result()
                    summary["players_processed"] += 1
                    summary["battles_seen"] += seen
                    accepted = deduplicator.keep_new(records)
                    summary["battles_accepted"] += len(accepted)
                    buffer.extend(accepted)
                    if snowball:
                        _enqueue(queue, queued, opponents, max_players, summary)
                    if len(buffer) >= shard_size:
                        flush()
                    progress.update(1)
                    progress.set_postfix(
                        accepted=summary["battles_accepted"], queued=len(queue)
                    )
                fill()
        flush()
    finally:
        progress.close()
        deduplicator.close()

    summary["requests"] = client.requests
    summary["rate_limited"] = client.rate_limited
    summary["request_errors"] = client.errors
    summary["bucket"] = bucket if upload else None
    summary["raw_dir"] = str(raw_dir)
    return summary


def _enqueue(
    queue: deque[str],
    queued: set[str],
    tags: Iterable[str],
    max_players: int | None,
    summary: dict[str, Any],
) -> None:
    if max_players is not None and len(queued) >= max_players * 4:
        return
    for tag in tags:
        if tag and tag not in queued:
            queued.add(tag)
            queue.append(tag)
