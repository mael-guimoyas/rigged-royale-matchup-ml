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
import random
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from dotenv import load_dotenv
from tqdm import tqdm

from .config import AppConfig
from .domain import _optional_int, parse_battle_row
from .extraction import SCHEMA, Deduplicator

DEFAULT_BASE_URL = "https://proxy.royaleapi.dev/v1"
# The RoyaleAPI proxy sits behind Cloudflare, which rejects the default
# "Python-urllib/x" User-Agent with error 1010. A normal UA is required.
USER_AGENT = "rigged-royale-matchup-ml/0.1 (+https://github.com/riggedroyale)"
CR_API_TOKEN_ENVS = ("CR_API_TOKEN", "CR_API_TOKEN2")
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


def _clean_env_token(value: str | None) -> str:
    return (value or "").strip().strip("\"'").replace(" ", "")


def _resolve_cr_api_tokens(token_mode: str = "1") -> list[tuple[str, str]]:
    normalized = token_mode.strip().lower()
    if normalized == "1":
        env_names = (CR_API_TOKEN_ENVS[0],)
    elif normalized == "2":
        env_names = (CR_API_TOKEN_ENVS[1],)
    elif normalized == "both":
        env_names = CR_API_TOKEN_ENVS
    else:
        raise ValueError("api token mode must be one of: 1, 2, both")

    tokens: list[tuple[str, str]] = []
    for env_name in env_names:
        token = _clean_env_token(os.getenv(env_name))
        if not token:
            raise RuntimeError(f"{env_name} is missing in .env")
        tokens.append((env_name, token))
    return tokens


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


def _opponent_candidates(battle: dict[str, Any]) -> list[tuple[str, str, int | None]]:
    """Snowball candidates carrying the trophy level they were discovered at.

    Returns ``(tag, mode_key, starting_trophies)`` for every participant. The
    trophy lets the frontier bucket a candidate *before* fetching it, so it can
    skip sub-5000 ladder players (we don't train on them) and steer the crawl
    toward under-filled trophy bands. Ranked (Path of Legends) battles hide
    trophies, so ranked candidates carry ``None`` and are always queued.
    """
    mode_key = mode_key_for(battle)
    out: list[tuple[str, str, int | None]] = []
    for participant in (battle.get("opponent") or []) + (battle.get("team") or []):
        tag = participant.get("tag")
        if tag:
            out.append(
                (normalize_tag(tag), mode_key, _optional_int(participant.get("startingTrophies")))
            )
    return out


def _ladder_bucket_label(trophies: int, buckets: list[int]) -> str | None:
    """Label the ladder trophy band for `trophies`, or None if above the top edge."""
    for lower, upper in zip(buckets, buckets[1:]):
        if lower <= trophies < upper:
            return f"ladder:{lower}-{upper - 1}"
    return None


def _segment_tracked(segment: str, min_trophies: int) -> bool:
    """Whether a parsed battle's segment is worth keeping for training.

    Keeps every ranked league and top-ladder-rank segment, plus ladder trophy
    bands at or above `min_trophies`. Drops sub-`min_trophies` ladder, plus the
    ``ladder:unknown`` / ``ladder:overflow`` junk -- "en dessous de 5000 on s'en
    fout" and unplaceable rows only dilute the corpus.
    """
    if segment.startswith("ranked:") or segment.startswith("ladder:top-"):
        return True
    if segment.startswith("ladder:"):
        lower = segment.split(":", 1)[1].split("-", 1)[0]
        return lower.isdigit() and int(lower) >= min_trophies
    return True


def _normalize_baseline(counts: Counter[str], buckets: list[int]) -> Counter[str]:
    """Re-label on-disk ladder bands onto the *current* trophy buckets.

    Old shards were written with coarser buckets (e.g. one ``ladder:12000-99998``
    blob before the 15000 seasonal split). Folding each legacy band onto the
    bucket holding its lower edge stops the balancer from treating a now-split
    band as empty and re-saturating the already-huge high-trophy region.
    """
    aligned: Counter[str] = Counter()
    for segment, count in counts.items():
        if segment.startswith("ladder:"):
            lower = segment.split(":", 1)[1].split("-", 1)[0]
            if lower.isdigit():
                label = _ladder_bucket_label(int(lower), buckets)
                aligned[label or segment] += count
                continue
        aligned[segment] += count
    return aligned


def _scan_existing_segment_counts(raw_dir: Path) -> Counter[str]:
    """Count battles per segment across existing shards to find current gaps.

    This is the "verifie ou il manque des combats" step: the frontier seeds its
    balance targets from what is already on disk, so a restart resumes filling
    the starved bands instead of re-piling onto the saturated high-trophy core.
    """
    counts: Counter[str] = Counter()
    for path in sorted(raw_dir.glob("*.parquet")):
        try:
            table = pq.read_table(path, columns=["segment"])
        except Exception:
            continue
        value_counts = pc.value_counts(table.column("segment").combine_chunks())
        for entry in value_counts:
            segment = entry["values"].as_py()
            if segment is not None:
                counts[str(segment)] += int(entry["counts"].as_py())
    return counts


class BalancedFrontier:
    """Trophy-aware crawl frontier that steers fetches toward starved segments.

    Snowballed opponents are bucketed by the trophy band they were discovered at
    (ladder) or pooled as ``ranked`` (Path of Legends hides trophies). `next_tag`
    always serves the *neediest* tracked band first -- the one with the fewest
    battles accepted so far (seeded from existing shards) -- so request budget is
    spent where data is missing instead of re-walking the saturated 12000+ core
    that produced ~1h of near-zero `accepted`.

    Ladder candidates below `min_trophies` are dropped outright: queueing them
    only bloats the queue and burns requests on duplicates we'd discard anyway.
    """

    def __init__(
        self,
        seeds: list[str],
        buckets: list[int],
        min_trophies: int,
        baseline_counts: Counter[str],
        max_queued: int | None,
    ) -> None:
        self._buckets = buckets
        self._min_trophies = min_trophies
        self._max_queued = max_queued
        self._seed_queue: deque[str] = deque(seeds)
        self._queues: dict[str, deque[str]] = {}
        self._queued: set[str] = set(seeds)
        self._counts: Counter[str] = Counter(baseline_counts)
        self._tracked = ["ranked"] + [
            f"ladder:{lower}-{upper - 1}"
            for lower, upper in zip(buckets, buckets[1:])
            if lower >= min_trophies
        ]

    def _score(self, bucket: str) -> int:
        if bucket == "ranked":
            return sum(value for key, value in self._counts.items() if key.startswith("ranked:"))
        return self._counts.get(bucket, 0)

    def _classify(self, mode_key: str, trophies: int | None) -> str | None:
        if mode_key == "ranked":
            return "ranked"
        if trophies is None or trophies < self._min_trophies:
            return None
        return _ladder_bucket_label(trophies, self._buckets)

    def record_accept(self, segment: str) -> None:
        self._counts[segment] += 1

    def add(self, candidates: list[tuple[str, str, int | None]]) -> int:
        """Queue new candidates by prospective band. Returns count skipped as low."""
        skipped = 0
        if self._max_queued is not None and len(self._queued) >= self._max_queued:
            return skipped
        for tag, mode_key, trophies in candidates:
            if not tag or tag in self._queued:
                continue
            bucket = self._classify(mode_key, trophies)
            if bucket is None:
                skipped += 1
                continue
            self._queued.add(tag)
            self._queues.setdefault(bucket, deque()).append(tag)
        return skipped

    def next_tag(self) -> str | None:
        # Serve the neediest band first; if its queue is dry, spend a seed to
        # discover more players (likely feeding deficits) before falling through.
        for bucket in sorted(self._tracked, key=self._score):
            queue = self._queues.get(bucket)
            if queue:
                return queue.popleft()
            if self._seed_queue:
                return self._seed_queue.popleft()
        if self._seed_queue:
            return self._seed_queue.popleft()
        for queue in self._queues.values():
            if queue:
                return queue.popleft()
        return None

    def queue_size(self) -> int:
        return len(self._seed_queue) + sum(len(queue) for queue in self._queues.values())


class ClashRoyaleClient:
    def __init__(
        self,
        limiter: RateLimiter | None = None,
        max_retries: int = 4,
        token_mode: str = "1",
        requests_per_second: float = 30.0,
    ) -> None:
        self._tokens = _resolve_cr_api_tokens(token_mode)
        self._token_index = 0
        self._base_url = (os.getenv("CR_API_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        shared_limiter = limiter or RateLimiter(requests_per_second)
        self._limiters = {name: shared_limiter for name, _ in self._tokens}
        self._max_retries = max_retries
        self._stats_lock = threading.Lock()
        self._token_stats: dict[str, dict[str, int]] = {
            name: {"requests": 0, "rate_limited": 0, "errors": 0}
            for name, _ in self._tokens
        }
        self.requests = 0
        self.rate_limited = 0
        self.errors = 0

    @property
    def token_names(self) -> list[str]:
        return [name for name, _ in self._tokens]

    @property
    def token_stats(self) -> dict[str, dict[str, int]]:
        with self._stats_lock:
            return {name: stats.copy() for name, stats in self._token_stats.items()}

    def _next_token(self) -> tuple[str, str]:
        with self._stats_lock:
            token = self._tokens[self._token_index]
            self._token_index = (self._token_index + 1) % len(self._tokens)
            return token

    def _bump(self, field: str, token_name: str | None = None) -> None:
        with self._stats_lock:
            setattr(self, field, getattr(self, field) + 1)
            if token_name is not None:
                self._token_stats[token_name][field] += 1

    def _get(self, path: str) -> Any | None:
        for attempt in range(self._max_retries):
            token_name, token = self._next_token()
            request = urllib.request.Request(
                f"{self._base_url}{path}",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/json",
                    "User-Agent": USER_AGENT,
                },
            )
            self._limiters[token_name].acquire()
            self._bump("requests", token_name)
            try:
                with urllib.request.urlopen(request, timeout=20) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as error:
                if error.code == 404:
                    return None
                if error.code == 429:
                    self._bump("rate_limited", token_name)
                    if attempt < self._max_retries - 1:
                        # Fastest legal retry: honor the API's own
                        # x-ratelimit-retry-after (microseconds) so we wake the
                        # instant the window reopens -- no wasted backoff. Fall
                        # back to a tiny 50ms pause if the header is absent or
                        # unparseable; clamp to 5s so a bogus value can't stall.
                        retry_after = (
                            error.headers.get("x-ratelimit-retry-after")
                            if error.headers
                            else None
                        )
                        try:
                            delay = float(retry_after) / 1_000_000 if retry_after else 0.05
                        except (TypeError, ValueError):
                            delay = 0.05
                        time.sleep(min(delay, 5.0))
                        continue
                if error.code in (500, 502, 503, 504) and attempt < self._max_retries - 1:
                    time.sleep(2.0 * (attempt + 1))
                    continue
                self._bump("errors", token_name)
                raise
            except (urllib.error.URLError, TimeoutError):
                if attempt < self._max_retries - 1:
                    time.sleep(2.0 * (attempt + 1))
                    continue
                self._bump("errors", token_name)
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
        # The seed query has no `where tracked` filter, so it can't use the partial
        # players_tracked_refresh index and instead does a full seq scan + top-N sort
        # over all rows. On throttled Supabase compute that scan alone is ~20s, which
        # blows the default role statement_timeout at large limits. Raise it for this
        # read-only session. The real fix is a covering index on
        # (last_analyzed_at desc nulls last) include (tag) -- see notes in README.
        connection.execute("set statement_timeout = '300s'")
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
    min_trophies: int,
) -> tuple[list[dict[str, Any]], list[tuple[str, str, int | None]], int]:
    """Network worker: battlelog first, profile only if a ranked battle exists.

    Skipping the profile fetch for ladder-only players roughly halves the request
    count. Returns parsed records (sub-`min_trophies` ladder dropped), snowball
    candidates with their discovery trophies, and battles seen. No dedup or file
    writes happen here so it is safe to run on many threads.
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
    candidates: list[tuple[str, str, int | None]] = []
    for battle in battles:
        if snowball:
            candidates.extend(_opponent_candidates(battle))
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
        if parsed is not None and _segment_tracked(parsed["segment"], min_trophies):
            records.append(parsed)
    return records, candidates, len(battles)


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
    min_trophies: int | None = None,
    balance: bool = True,
    api_token_mode: str = "1",
    show_progress: bool = True,
    stats_interval_seconds: float = 10.0,
) -> dict[str, Any]:
    load_dotenv(config.resolve(".env"))
    seeds = _seed_tags(config, tags_file, from_supabase)
    if not seeds:
        raise RuntimeError("No seed tags. Pass --tags-file or --from-supabase.")
    # Randomize the crawl entry order so repeated runs over the same seed file
    # explore a different region first instead of re-walking the identical BFS
    # tree (which, when capped by max_players/max_battles, just re-collects the
    # same battles). `random` is auto-seeded from OS entropy per process, so each
    # invocation diverges.
    random.shuffle(seeds)

    raw_dir = config.resolve(config.data["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    client = ClashRoyaleClient(
        token_mode=api_token_mode,
        requests_per_second=requests_per_second,
    )
    effective_workers = workers
    uploader = StorageClient(bucket, prefix, create=True) if upload else None
    deduplicator = Deduplicator(config.resolve(config.data["dedup_db"]))
    allowed_modes = set(config.data.get("allowed_modes") or [])

    trophy_buckets = [int(edge) for edge in config.data["trophy_buckets"]]
    if min_trophies is None:
        min_trophies = int(config.data.get("collect_min_trophies", 5000))
    # Baseline = where battles already are, so balance fills the gaps on disk and
    # a restart resumes the starved bands instead of re-piling onto the core.
    baseline_counts = (
        _normalize_baseline(_scan_existing_segment_counts(raw_dir), trophy_buckets)
        if balance
        else Counter()
    )
    max_queued = max_players * 4 if max_players is not None else None
    frontier = BalancedFrontier(seeds, trophy_buckets, min_trophies, baseline_counts, max_queued)

    shard_index = _next_shard_index(raw_dir)
    buffer: list[dict[str, Any]] = []
    summary = {
        "players_processed": 0,
        "battles_seen": 0,
        "battles_accepted": 0,
        "shards_written": 0,
        "uploaded": 0,
        "seed_tags": len(seeds),
        "candidates_skipped_low": 0,
        "min_trophies": min_trophies,
        "balance": balance,
        "api_token_mode": api_token_mode,
        "api_token_envs": client.token_names,
        "requests_per_second_total": requests_per_second,
        "effective_workers": effective_workers,
        "progress": show_progress,
        "stats_interval_seconds": stats_interval_seconds,
    }
    run_by_segment: Counter[str] = Counter()
    progress = tqdm(desc="CR API players", unit="player", disable=not show_progress)
    if show_progress and balance and baseline_counts:
        top = ", ".join(
            f"{seg}={count:,}"
            for seg, count in sorted(baseline_counts.items(), key=lambda kv: -kv[1])[:6]
        )
        progress.write(
            f"Baseline {sum(baseline_counts.values()):,} battles on disk; "
            f"steering toward starved >= {min_trophies} bands. Top: {top}"
        )

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
    target_inflight = max(effective_workers * 2, effective_workers + 4)
    submitted = 0
    started_at = time.perf_counter()
    last_stats_at = started_at
    last_stats_requests = 0
    last_stats_players = 0

    def can_submit() -> bool:
        if max_players is not None and submitted >= max_players:
            return False
        if max_battles is not None and summary["battles_accepted"] >= max_battles:
            return False
        return frontier.queue_size() > 0

    def maybe_emit_status() -> None:
        nonlocal last_stats_at, last_stats_requests, last_stats_players
        if show_progress or stats_interval_seconds <= 0:
            return
        now = time.perf_counter()
        interval = now - last_stats_at
        if interval < stats_interval_seconds:
            return
        elapsed = max(now - started_at, 1e-9)
        total_rps = client.requests / elapsed
        recent_rps = (client.requests - last_stats_requests) / max(interval, 1e-9)
        recent_players = (
            summary["players_processed"] - last_stats_players
        ) / max(interval, 1e-9)
        print(
            "CR API "
            f"elapsed={elapsed:,.0f}s "
            f"players={summary['players_processed']:,} "
            f"players/s={recent_players:.2f} "
            f"requests={client.requests:,} "
            f"req/s={recent_rps:.2f} "
            f"avg_req/s={total_rps:.2f} "
            f"accepted={summary['battles_accepted']:,} "
            f"queued={frontier.queue_size():,} "
            f"429={client.rate_limited:,}",
            file=sys.stderr,
            flush=True,
        )
        last_stats_at = now
        last_stats_requests = client.requests
        last_stats_players = summary["players_processed"]

    try:
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            inflight: dict[Future, str] = {}

            def fill() -> None:
                nonlocal submitted
                while can_submit() and len(inflight) < target_inflight:
                    tag = frontier.next_tag()
                    if tag is None:
                        break
                    future = executor.submit(
                        _fetch_player,
                        client,
                        tag,
                        config.data,
                        allowed_modes,
                        snowball,
                        min_trophies,
                    )
                    inflight[future] = tag
                    submitted += 1

            fill()
            while inflight:
                done, _ = wait(list(inflight), return_when=FIRST_COMPLETED)
                for future in done:
                    inflight.pop(future)
                    records, candidates, seen = future.result()
                    summary["players_processed"] += 1
                    summary["battles_seen"] += seen
                    accepted = deduplicator.keep_new(records)
                    summary["battles_accepted"] += len(accepted)
                    buffer.extend(accepted)
                    for record in accepted:
                        frontier.record_accept(record["segment"])
                        run_by_segment[record["segment"]] += 1
                    if snowball:
                        summary["candidates_skipped_low"] += frontier.add(candidates)
                    if len(buffer) >= shard_size:
                        flush()
                    if show_progress:
                        progress.update(1)
                        progress.set_postfix(
                            accepted=summary["battles_accepted"],
                            queued=frontier.queue_size(),
                        )
                    maybe_emit_status()
                fill()
        flush()
    finally:
        progress.close()
        deduplicator.close()

    summary["requests"] = client.requests
    summary["rate_limited"] = client.rate_limited
    summary["request_errors"] = client.errors
    summary["api_token_stats"] = client.token_stats
    summary["bucket"] = bucket if upload else None
    summary["raw_dir"] = str(raw_dir)
    summary["accepted_by_segment"] = dict(run_by_segment)
    return summary
