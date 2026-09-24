#!/usr/bin/env python3
"""Watch Lambda Cloud for a 1x Blackwell GPU instance and launch it the moment one frees up.

Lambda's single-GPU Blackwell (B200) capacity is almost never visible: it appears
for a few seconds when someone else terminates, then it is gone. So this polls
`/instance-types` and, the instant a target type reports capacity in an allowed
region, fires `/instance-operations/launch`. Losing the race returns
`insufficient-capacity`, which is expected and simply resumes polling.

Usage (see --help for everything):

    export LAMBDA_API_KEY=secret_...
    tools/lambda_watch.py --list                                # account + availability
    tools/lambda_watch.py --ssh-key mosaicist-h100 --dry-run     # watch, launch nothing
    tools/lambda_watch.py --ssh-key mosaicist-h100 --say         # arm it for real

Stdlib only, so it runs on any Python 3.10+ host without a virtualenv.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import random
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

API_BASE = "https://cloud.lambda.ai/api/v1"
USER_AGENT = "mosaicist-lambda-watch/1.0"

# Lambda documents 1 req/s overall and 1 per 12s (5/min) for the launch endpoint.
# Staying a hair under the documented floor keeps us off the 429 path.
MIN_LAUNCH_INTERVAL_S = 13.0
MIN_REQUEST_INTERVAL_S = 1.1

# A launch can take well over the default 30s to answer. Timing out early turns
# a real success into an "unknown" outcome, so give that one call more room.
LAUNCH_TIMEOUT_S = 120.0

# After a launch whose outcome is unclear (timeout, 5xx, garbled response), how
# long to watch /instances for the new box before deciding it never happened.
RECONCILE_BUDGET_S = 180.0
RECONCILE_INTERVAL_S = 15.0

# GPU marketing names in the Blackwell family, matched against `gpu_description`
# so instance types Lambda adds later (GB200, B300, RTX PRO 6000) are picked up
# without a code change.
BLACKWELL_GPU_PATTERN = (
    r"\b(?:B100|B200|B300|GB200|GB300|RTX\s*(?:PRO\s*)?6000\s*(?:Blackwell|SE)|Blackwell)\b"
)

INSUFFICIENT_CAPACITY = "instance-operations/launch/insufficient-capacity"

# Error codes taken from the published OpenAPI spec. "Retryable" means waiting
# could plausibly fix it; everything in FATAL_CODES needs a human.
RETRYABLE_CODES = {
    INSUFFICIENT_CAPACITY,
    "global/internal-error",
    "global/conflict",
}
FATAL_CODES = {
    "global/account-inactive",
    "global/forbidden",
    "global/invalid-address",
    "global/invalid-api-key",
    "global/invalid-parameters",
    "global/not-found",
    "global/object-does-not-exist",
    "global/quota-exceeded",
    "instance-operations/launch/file-system-in-wrong-region",
}


class ApiError(Exception):
    """A structured Lambda API error.

    Deliberately a plain class rather than a dataclass: @dataclass on an
    Exception subclass interacts badly with Exception.args and needs the module
    registered in sys.modules, which breaks when this file is loaded by path.
    """

    def __init__(
        self,
        status: int,
        code: str,
        message: str,
        suggestion: str = "",
        request_id: str = "",
    ) -> None:
        super().__init__(status, code, message)
        self.status = status
        self.code = code
        self.message = message
        self.suggestion = suggestion
        self.request_id = request_id

    def __str__(self) -> str:
        bits = [f"HTTP {self.status}", self.code or "?", self.message or ""]
        if self.suggestion:
            bits.append(f"({self.suggestion})")
        if self.request_id:
            bits.append(f"[req {self.request_id}]")
        return " ".join(b for b in bits if b)

    @property
    def retryable(self) -> bool:
        if self.code in RETRYABLE_CODES:
            return True
        if self.code in FATAL_CODES:
            return False
        # An unknown 4xx is treated as fatal so we do not hammer a request that
        # can never succeed; an unknown 5xx or a 429 is transient.
        return self.status >= 500 or self.status == 429


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Candidate:
    """One (instance type, region) pair that currently reports free capacity."""

    instance_type: str
    region: str
    gpu_description: str
    gpus: int
    price_cents_per_hour: int

    @property
    def price_str(self) -> str:
        return f"${self.price_cents_per_hour / 100:.2f}/hr"

    def __str__(self) -> str:
        return f"{self.instance_type} in {self.region} ({self.gpu_description}, {self.price_str})"


def matches_target(
    type_name: str,
    gpu_description: str,
    gpus: int,
    *,
    gpu_count: int,
    gpu_pattern: str,
    explicit_types: list[str] | None,
) -> bool:
    """Is this instance type one we are hunting for?

    `explicit_types` short-circuits everything else, so `--instance-type` can pin
    an exact name (including a multi-GPU one) without fighting the family filter.
    """
    if explicit_types:
        return type_name in explicit_types
    if gpus != gpu_count:
        return False
    return re.search(gpu_pattern, gpu_description, re.IGNORECASE) is not None


def find_candidates(
    instance_types_payload: dict,
    *,
    gpu_count: int,
    gpu_pattern: str,
    explicit_types: list[str] | None,
    allowed_regions: list[str] | None,
    max_price_cents: int | None,
) -> list[Candidate]:
    """Extract every (type, region) with capacity that passes all filters.

    Ordered by region preference when `allowed_regions` is given, then by price,
    so the cheapest acceptable box in the most-preferred region is tried first.
    """
    out: list[Candidate] = []
    for type_name, entry in (instance_types_payload or {}).items():
        it = entry.get("instance_type") or {}
        specs = it.get("specs") or {}
        gpus = specs.get("gpus", 0)
        gpu_desc = it.get("gpu_description") or ""
        if not matches_target(
            type_name,
            gpu_desc,
            gpus,
            gpu_count=gpu_count,
            gpu_pattern=gpu_pattern,
            explicit_types=explicit_types,
        ):
            continue
        price = it.get("price_cents_per_hour", 0)
        if max_price_cents is not None and price > max_price_cents:
            continue
        for region in entry.get("regions_with_capacity_available") or []:
            name = region.get("name") if isinstance(region, dict) else region
            if not name:
                continue
            if allowed_regions and name not in allowed_regions:
                continue
            out.append(
                Candidate(
                    instance_type=type_name,
                    region=name,
                    gpu_description=gpu_desc,
                    gpus=gpus,
                    price_cents_per_hour=price,
                )
            )

    def sort_key(c: Candidate) -> tuple[int, int, str]:
        rank = allowed_regions.index(c.region) if allowed_regions else 0
        return (rank, c.price_cents_per_hour, c.instance_type)

    return sorted(out, key=sort_key)


def matching_target_types(
    instance_types_payload: dict,
    *,
    gpu_count: int,
    gpu_pattern: str,
    explicit_types: list[str] | None,
) -> list[str]:
    """Every instance type name we would accept, regardless of capacity."""
    names = []
    for type_name, entry in (instance_types_payload or {}).items():
        it = entry.get("instance_type") or {}
        if matches_target(
            type_name,
            it.get("gpu_description") or "",
            (it.get("specs") or {}).get("gpus", 0),
            gpu_count=gpu_count,
            gpu_pattern=gpu_pattern,
            explicit_types=explicit_types,
        ):
            names.append(type_name)
    return sorted(names)


def existing_target_instances(instances: list[dict], target_types: list[str]) -> list[dict]:
    """Instances we already own of a target type, in any non-terminal state.

    Keeps a restarted watcher from quietly launching a second $7/hr box next to
    the one it launched an hour ago.
    """
    live = {"active", "booting", "unhealthy"}
    out = []
    for inst in instances or []:
        name = ((inst.get("instance_type") or {}).get("name")) or ""
        if name in target_types and (inst.get("status") or "") in live:
            out.append(inst)
    return out


def new_target_instances(
    instances: list[dict], known_ids: set[str], target_types: list[str]
) -> list[dict]:
    """Live target instances that were not on the account when we started.

    This is the ground truth for "did our launch land?". The launch response can
    be lost (timeout, dropped connection) or look like a failure (5xx after the
    backend already committed), but the instance still shows up here.
    """
    return [
        inst
        for inst in existing_target_instances(instances, target_types)
        if inst.get("id") not in known_ids
    ]


def next_sleep(interval: float, jitter: float) -> float:
    """Poll interval with jitter, so many watchers do not sync into one burst."""
    return max(1.0, interval + random.uniform(-jitter, jitter))


def parse_duration(text: str) -> float:
    """Parse `90`, `30s`, `15m`, `6h`, `1d` into seconds."""
    cleaned = str(text).strip().lower()
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([smhd]?)", cleaned)
    if not m:
        raise SystemExit(f"cannot parse duration {text!r}; use forms like 30s, 15m, 6h")
    return float(m.group(1)) * {"": 1, "s": 1, "m": 60, "h": 3600, "d": 86400}[m.group(2)]


def _first_key_line(text: str) -> str:
    """Pull the key out of a file that may also hold labels or blank lines."""
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("secret_"):
            return line
    stripped = text.strip()
    if not stripped:
        raise SystemExit("API key file is empty")
    return stripped.splitlines()[0].strip()


# --------------------------------------------------------------------------- #
# API client
# --------------------------------------------------------------------------- #


class LambdaClient:
    def __init__(self, api_key: str, *, base: str = API_BASE, timeout: float = 30.0):
        self._auth = base64.b64encode(f"{api_key}:".encode()).decode()
        self._base = base.rstrip("/")
        self._timeout = timeout
        self._last_request = 0.0

    def _throttle(self) -> None:
        gap = time.monotonic() - self._last_request
        if self._last_request and gap < MIN_REQUEST_INTERVAL_S:
            time.sleep(MIN_REQUEST_INTERVAL_S - gap)
        self._last_request = time.monotonic()

    def request(
        self, method: str, path: str, body: dict | None = None, *, timeout: float | None = None
    ) -> dict:
        self._throttle()
        data = json.dumps(body).encode() if body is not None else None
        headers = {
            "Authorization": f"Basic {self._auth}",
            "Accept": "application/json",
            # Cloudflare in front of cloud.lambda.ai 403s urllib's default
            # "Python-urllib/3.x" agent (error 1010), so send a real one.
            "User-Agent": USER_AGENT,
        }
        if data:
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(
            f"{self._base}/{path.lstrip('/')}", data=data, method=method, headers=headers
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout or self._timeout) as resp:
                return json.loads(resp.read() or b"{}")
        except urllib.error.HTTPError as exc:
            raw = exc.read()
            try:
                err = (json.loads(raw) or {}).get("error") or {}
            except (ValueError, TypeError):
                err = {}
            raise ApiError(
                status=exc.code,
                code=err.get("code", ""),
                message=err.get("message") or raw[:200].decode("utf-8", "replace"),
                suggestion=err.get("suggestion", ""),
                request_id=err.get("request_id", ""),
            ) from None

    def instance_types(self) -> dict:
        return self.request("GET", "/instance-types").get("data") or {}

    def instances(self) -> list[dict]:
        return self.request("GET", "/instances").get("data") or []

    def instance(self, instance_id: str) -> dict:
        return self.request("GET", f"/instances/{instance_id}").get("data") or {}

    def ssh_keys(self) -> list[dict]:
        return self.request("GET", "/ssh-keys").get("data") or []

    def launch(self, payload: dict) -> list[str]:
        data = (
            self.request(
                "POST", "/instance-operations/launch", payload, timeout=LAUNCH_TIMEOUT_S
            ).get("data")
            or {}
        )
        return data.get("instance_ids") or []


# --------------------------------------------------------------------------- #
# Output / notification
# --------------------------------------------------------------------------- #


def log(msg: str, *, stream=sys.stdout) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", file=stream, flush=True)


def notify(title: str, message: str, *, webhook: str | None, say: bool) -> None:
    """Best-effort desktop / webhook alert.

    Never raises: a failed ping must not kill a watcher that just won a GPU.
    """
    if sys.platform == "darwin":
        script = (
            f"display notification {json.dumps(message)} "
            f'with title {json.dumps(title)} sound name "Hero"'
        )
        try:
            subprocess.run(
                ["osascript", "-e", script], timeout=10, capture_output=True, check=False
            )
            if say:
                subprocess.run(["say", message], timeout=30, capture_output=True, check=False)
        except (OSError, subprocess.SubprocessError) as exc:
            log(f"notify: desktop alert failed: {exc}")
    if webhook:
        try:
            req = urllib.request.Request(
                webhook,
                data=json.dumps({"text": f"{title}: {message}"}).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=15):
                pass
        except (urllib.error.URLError, OSError) as exc:
            log(f"notify: webhook failed: {exc}")


# --------------------------------------------------------------------------- #
# Watcher
# --------------------------------------------------------------------------- #


@dataclass
class Config:
    api_key: str
    ssh_keys: list[str]
    gpu_count: int = 1
    gpu_pattern: str = BLACKWELL_GPU_PATTERN
    instance_types: list[str] = field(default_factory=list)
    regions: list[str] = field(default_factory=list)
    max_price_cents: int | None = None
    interval: float = 30.0
    jitter: float = 5.0
    name: str | None = None
    image_family: str | None = None
    file_systems: list[str] = field(default_factory=list)
    user_data_file: str | None = None
    dry_run: bool = False
    once: bool = False
    timeout_s: float | None = None
    allow_duplicate: bool = False
    wait_for_active: bool = True
    webhook: str | None = None
    say: bool = False


_stop = False


def _install_signal_handlers() -> None:
    def handler(signum, _frame):
        global _stop
        _stop = True
        log(f"received signal {signum}; stopping after current check")

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, handler)


def build_launch_payload(cfg: Config, cand: Candidate) -> dict:
    payload: dict = {
        "region_name": cand.region,
        "instance_type_name": cand.instance_type,
        "ssh_key_names": cfg.ssh_keys,
    }
    if cfg.file_systems:
        payload["file_system_names"] = cfg.file_systems
    if cfg.name:
        payload["name"] = cfg.name
    if cfg.image_family:
        payload["image"] = {"family": cfg.image_family}
    if cfg.user_data_file:
        with open(os.path.expanduser(cfg.user_data_file), encoding="utf-8") as fh:
            payload["user_data"] = fh.read()
    return payload


def wait_until_active(client: LambdaClient, instance_id: str, *, budget_s: float = 900.0) -> dict:
    """Poll one instance until it leaves `booting`. Returns the last payload seen."""
    deadline = time.monotonic() + budget_s
    inst: dict = {}
    while time.monotonic() < deadline and not _stop:
        try:
            inst = client.instance(instance_id)
        except ApiError as exc:
            log(f"  status check failed ({exc}); retrying")
            time.sleep(10)
            continue
        status = inst.get("status", "?")
        log(f"  instance {instance_id}: status={status} ip={inst.get('ip') or '-'}")
        if status != "booting":
            return inst
        time.sleep(15)
    return inst


def report_success(inst: dict, instance_id: str, cand: Candidate, cfg: Config) -> None:
    ip = inst.get("ip") or ""
    status = inst.get("status", "launch accepted")
    log("=" * 70)
    log(f"LAUNCHED {cand.instance_type} in {cand.region} at {cand.price_str}")
    log(f"  id     : {instance_id}")
    log(f"  status : {status}")
    if ip:
        log(f"  ip     : {ip}")
        log(f"  ssh    : ssh ubuntu@{ip}")
    if inst.get("jupyter_url"):
        log(f"  jupyter: {inst['jupyter_url']}")
    log(f"  billing: accruing at {cand.price_str} until you terminate it")
    log("=" * 70)
    notify(
        "Lambda GPU acquired",
        f"{cand.instance_type} in {cand.region} is {status}" + (f" at {ip}" if ip else ""),
        webhook=cfg.webhook,
        say=cfg.say,
    )


def _preflight(cfg: Config, client: LambdaClient) -> tuple[dict, list[str]] | int:
    """Validate the key, the SSH key names, and the filters before the poll loop.

    Returns (instance_types_payload, target_type_names), or an exit code. Doing
    this up front means a typo surfaces now, not at the one moment capacity
    appears.
    """
    try:
        key_names = {k["name"] for k in client.ssh_keys()}
    except ApiError as exc:
        log(f"cannot reach Lambda API: {exc}", stream=sys.stderr)
        return 2
    missing = [k for k in cfg.ssh_keys if k not in key_names]
    if missing:
        log(
            f"ssh key(s) not found on this account: {', '.join(missing)}; "
            f"available: {', '.join(sorted(key_names)) or '(none)'}",
            stream=sys.stderr,
        )
        return 2

    try:
        types = client.instance_types()
    except ApiError as exc:
        log(f"cannot list instance types: {exc}", stream=sys.stderr)
        return 2

    targets = matching_target_types(
        types,
        gpu_count=cfg.gpu_count,
        gpu_pattern=cfg.gpu_pattern,
        explicit_types=cfg.instance_types,
    )
    if not targets:
        log(
            "no instance type matches the filters "
            f"(gpu_count={cfg.gpu_count}, pattern={cfg.gpu_pattern!r}, "
            f"types={cfg.instance_types or 'any'})",
            stream=sys.stderr,
        )
        return 2

    log(f"watching for: {', '.join(targets)}")
    log(f"ssh key(s): {', '.join(cfg.ssh_keys) or '(none)'}")
    if cfg.regions:
        log(f"regions (in preference order): {', '.join(cfg.regions)}")
    if cfg.max_price_cents is not None:
        log(f"price ceiling: ${cfg.max_price_cents / 100:.2f}/hr")
    if cfg.dry_run:
        log("DRY RUN: will report capacity but never launch")
    return types, targets


def _find_landed(
    client: LambdaClient,
    known_ids: set[str],
    targets: list[str],
    *,
    budget_s: float,
) -> dict | None | str:
    """Look for an instance our launch created, polling for up to `budget_s`.

    Returns the instance if found, None if the account was listed and nothing
    new is there, or 'unknown' if /instances could never be read.
    """
    deadline = time.monotonic() + budget_s
    listed = False
    while True:
        try:
            fresh = new_target_instances(client.instances(), known_ids, targets)
            listed = True
            if fresh:
                return fresh[0]
        except (ApiError, urllib.error.URLError, OSError, ValueError) as exc:
            log(f"  cannot list instances to check the launch ({exc})")
        if time.monotonic() >= deadline or _stop:
            return None if listed else "unknown"
        time.sleep(RECONCILE_INTERVAL_S)


def _try_launch(
    cfg: Config,
    client: LambdaClient,
    cand: Candidate,
    known_ids: set[str],
    targets: list[str],
) -> tuple[str, dict] | str:
    """One launch attempt. Returns (instance_id, instance) on success, else a
    disposition string: 'miss' (keep hunting), 'fatal', or 'uncertain' (a
    launch may have landed and we cannot confirm either way, so stop).

    Only a response carrying an instance id counts as success on its own. Every
    other outcome is checked against /instances before we resume hunting,
    because a timeout or 5xx can hide a launch that went through, and hunting
    on after that would leave a paid instance idling and launch a second one.
    """
    log(f"  launching {cand} ...")
    ids: list[str] = []
    clean_miss = False
    try:
        ids = client.launch(build_launch_payload(cfg, cand))
        if not ids:
            log("  launch answered without an instance id; checking whether it landed")
    except ApiError as exc:
        if exc.code == INSUFFICIENT_CAPACITY:
            log("  lost the race (insufficient capacity)")
            clean_miss = True
        elif not exc.retryable:
            log(f"  fatal launch error: {exc}", stream=sys.stderr)
            notify("Lambda watcher stopped", str(exc), webhook=cfg.webhook, say=cfg.say)
            return "fatal"
        else:
            log(f"  launch failed ({exc}); checking whether it landed anyway")
    except (urllib.error.URLError, OSError, ValueError) as exc:
        log(f"  launch request errored in flight ({exc}); checking whether it landed")

    if ids:
        instance_id = ids[0]
        log(f"  launch accepted: {instance_id}")
    else:
        # A clean insufficient-capacity is near-certain, so one look is enough;
        # anything murkier gets a few minutes for the instance to show up.
        found = _find_landed(
            client, known_ids, targets, budget_s=0 if clean_miss else RECONCILE_BUDGET_S
        )
        if found == "unknown" and not clean_miss:
            msg = (
                f"launch of {cand} may have succeeded but /instances cannot be read; "
                "stopped so it cannot launch twice. Check the dashboard."
            )
            log(f"  {msg}", stream=sys.stderr)
            notify("Lambda watcher stopped", msg, webhook=cfg.webhook, say=cfg.say)
            return "uncertain"
        if not isinstance(found, dict):
            log("  no new instance on the account; still hunting")
            return "miss"
        instance_id = found["id"]
        log(f"  launch did land: found {instance_id} ({found.get('status')})")

    inst = wait_until_active(client, instance_id) if cfg.wait_for_active else {}
    return instance_id, inst


def watch(cfg: Config) -> int:
    client = LambdaClient(cfg.api_key)

    pre = _preflight(cfg, client)
    if isinstance(pre, int):
        return pre
    types, targets = pre

    # Snapshot what the account already has, so a launch whose response we
    # never saw can still be recognized by the new id it leaves behind.
    try:
        instances = client.instances()
    except ApiError as exc:
        log(f"cannot list instances: {exc}", stream=sys.stderr)
        return 2
    known_ids = {inst.get("id") for inst in instances}

    if not cfg.allow_duplicate:
        dupes = existing_target_instances(instances, targets)
        if dupes:
            for inst in dupes:
                log(
                    f"already own {inst['instance_type']['name']} {inst['id']} "
                    f"({inst.get('status')}) in {(inst.get('region') or {}).get('name')} "
                    f"at {inst.get('ip') or 'no ip yet'}"
                )
            log("nothing to do; pass --allow-duplicate to launch another anyway")
            return 0

    deadline = time.monotonic() + cfg.timeout_s if cfg.timeout_s else None
    last_launch_attempt = 0.0
    checks = 0
    backoff = 0.0
    fresh = True  # _preflight already fetched the first payload

    while not _stop:
        if deadline and time.monotonic() > deadline:
            log(f"timeout reached after {checks} checks; no capacity found")
            return 3

        if not fresh:
            try:
                types = client.instance_types()
            except ApiError as exc:
                if not exc.retryable:
                    log(f"fatal: {exc}", stream=sys.stderr)
                    return 2
                backoff = min(max(backoff * 2, 5.0), 120.0)
                log(f"poll failed ({exc}); backing off {backoff:.0f}s")
                time.sleep(backoff)
                continue
            except (urllib.error.URLError, OSError, ValueError) as exc:
                backoff = min(max(backoff * 2, 5.0), 120.0)
                log(f"poll error ({exc}); backing off {backoff:.0f}s")
                time.sleep(backoff)
                continue
        fresh = False
        backoff = 0.0
        checks += 1

        candidates = find_candidates(
            types,
            gpu_count=cfg.gpu_count,
            gpu_pattern=cfg.gpu_pattern,
            explicit_types=cfg.instance_types,
            allowed_regions=cfg.regions or None,
            max_price_cents=cfg.max_price_cents,
        )

        if not candidates:
            log(f"check #{checks}: no capacity")
        else:
            log(f"check #{checks}: CAPACITY -> {'; '.join(str(c) for c in candidates)}")
            for cand in candidates:
                if cfg.dry_run:
                    log(f"  dry run: would launch {cand}")
                    continue
                gap = time.monotonic() - last_launch_attempt
                if last_launch_attempt and gap < MIN_LAUNCH_INTERVAL_S:
                    time.sleep(MIN_LAUNCH_INTERVAL_S - gap)
                last_launch_attempt = time.monotonic()
                result = _try_launch(cfg, client, cand, known_ids, targets)
                if result == "fatal":
                    return 2
                if result == "uncertain":
                    return 4
                if result == "miss":
                    continue
                instance_id, inst = result
                report_success(inst, instance_id, cand, cfg)
                return 0

        if cfg.once:
            return 0 if candidates else 1
        time.sleep(next_sleep(cfg.interval, cfg.jitter))

    log("stopped")
    return 130


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def read_api_key(explicit: str | None, key_file: str | None) -> str:
    """Resolve the key from --api-key, --api-key-file, $LAMBDA_API_KEY, or
    ~/.lambda/api_key, in that order."""
    if explicit:
        return explicit.strip()
    if key_file:
        with open(os.path.expanduser(key_file), encoding="utf-8") as fh:
            return _first_key_line(fh.read())
    env = os.environ.get("LAMBDA_API_KEY")
    if env:
        return env.strip()
    default = os.path.expanduser("~/.lambda/api_key")
    if os.path.exists(default):
        with open(default, encoding="utf-8") as fh:
            return _first_key_line(fh.read())
    raise SystemExit(
        "no API key: pass --api-key/--api-key-file, set LAMBDA_API_KEY, "
        "or write the key to ~/.lambda/api_key"
    )


def parse_args(argv: list[str] | None = None) -> tuple[Config, argparse.Namespace]:
    p = argparse.ArgumentParser(
        description=(
            "Poll Lambda Cloud for a 1x Blackwell GPU instance and launch it when one appears."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  # what does the account look like right now?
  %(prog)s --list

  # watch for a 1x B200 but never launch, to sanity-check the filters
  %(prog)s --ssh-key mosaicist-h100 --dry-run

  # hunt indefinitely, alert loudly the moment it lands
  %(prog)s --ssh-key mosaicist-h100 --name mosaicist-b200 --say

  # US regions only, give up after 6 hours, refuse anything over $8/hr
  %(prog)s --ssh-key mosaicist-h100 --region us-west-1 --region us-east-1 \\
      --timeout 6h --max-price 8.00
""",
    )
    p.add_argument("--api-key", help="Lambda API key (prefer --api-key-file or $LAMBDA_API_KEY)")
    p.add_argument("--api-key-file", help="file containing the API key")
    p.add_argument(
        "--ssh-key",
        action="append",
        default=[],
        metavar="NAME",
        help="SSH key name already registered with Lambda (required to launch; repeatable)",
    )
    p.add_argument(
        "--instance-type",
        action="append",
        default=[],
        metavar="NAME",
        help=(
            "pin an exact instance type (e.g. gpu_1x_b200_sxm6); repeatable, "
            "overrides --gpu-count/--gpu-pattern"
        ),
    )
    p.add_argument("--gpu-count", type=int, default=1, help="GPUs per instance (default: 1)")
    p.add_argument(
        "--gpu-pattern",
        default=BLACKWELL_GPU_PATTERN,
        help="regex matched against gpu_description (default: the Blackwell family)",
    )
    p.add_argument(
        "--region",
        action="append",
        default=[],
        metavar="NAME",
        help="restrict to this region; repeat for a preference order (default: any)",
    )
    p.add_argument(
        "--max-price", type=float, metavar="USD", help="refuse to launch above this hourly price"
    )
    p.add_argument("--interval", default="30s", help="poll interval (default: 30s)")
    p.add_argument("--jitter", default="5s", help="random +/- added to the interval (default: 5s)")
    p.add_argument("--timeout", help="give up after this long (e.g. 6h, 90m); default: never")
    p.add_argument("--name", help="name for the launched instance")
    p.add_argument("--image-family", help="image family, e.g. lambda-stack-24-04")
    p.add_argument(
        "--file-system",
        action="append",
        default=[],
        metavar="NAME",
        help="filesystem to mount; repeatable",
    )
    p.add_argument("--user-data", metavar="FILE", help="cloud-init user data to run on first boot")
    p.add_argument("--dry-run", action="store_true", help="report capacity but never launch")
    p.add_argument("--once", action="store_true", help="check a single time and exit")
    p.add_argument(
        "--allow-duplicate",
        action="store_true",
        help="launch even if a matching instance is already running",
    )
    p.add_argument(
        "--no-wait",
        action="store_true",
        help="exit as soon as the launch is accepted, without waiting for boot",
    )
    p.add_argument(
        "--webhook", help='POST {"text": ...} here on success/fatal error (Slack-compatible)'
    )
    p.add_argument("--say", action="store_true", help="speak the alert out loud on macOS")
    p.add_argument(
        "--list", action="store_true", help="print instance types, availability, and account, then exit"
    )

    a = p.parse_args(argv)
    if not a.ssh_key and not (a.list or a.dry_run):
        p.error("--ssh-key is required to launch (use --list or --dry-run to look around first)")

    cfg = Config(
        api_key=read_api_key(a.api_key, a.api_key_file),
        ssh_keys=a.ssh_key,
        gpu_count=a.gpu_count,
        gpu_pattern=a.gpu_pattern,
        instance_types=a.instance_type,
        regions=a.region,
        max_price_cents=round(a.max_price * 100) if a.max_price is not None else None,
        interval=parse_duration(a.interval),
        jitter=parse_duration(a.jitter),
        name=a.name,
        image_family=a.image_family,
        file_systems=a.file_system,
        user_data_file=a.user_data,
        dry_run=a.dry_run,
        once=a.once,
        timeout_s=parse_duration(a.timeout) if a.timeout else None,
        allow_duplicate=a.allow_duplicate,
        wait_for_active=not a.no_wait,
        webhook=a.webhook,
        say=a.say,
    )
    return cfg, a


def cmd_list(cfg: Config) -> int:
    client = LambdaClient(cfg.api_key)
    try:
        types = client.instance_types()
        instances = client.instances()
        keys = client.ssh_keys()
    except ApiError as exc:
        log(f"API error: {exc}", stream=sys.stderr)
        return 2

    print(f"{'GPUs':>4}  {'instance type':<24} {'GPU':<26} {'$/hr':>8}  regions with capacity")
    rows = []
    for name, entry in types.items():
        it = entry["instance_type"]
        avail = ", ".join(r["name"] for r in entry["regions_with_capacity_available"]) or "-"
        rows.append(
            (
                (it["specs"]["gpus"], name),
                f"{it['specs']['gpus']:>4}  {name:<24} {it['gpu_description']:<26} "
                f"{it['price_cents_per_hour'] / 100:>8.2f}  {avail}",
            )
        )
    for _, line in sorted(rows):
        print(line)

    print(f"\nssh keys: {', '.join(k['name'] for k in keys) or '(none)'}")
    print("running instances:")
    if not instances:
        print("  (none)")
    for inst in instances:
        print(
            f"  {inst['id']}  {inst['instance_type']['name']:<20} "
            f"{str(inst.get('status')):<9} {(inst.get('region') or {}).get('name', '?'):<14} "
            f"{inst.get('ip') or '-':<16} ({inst.get('name') or 'unnamed'})"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    cfg, args = parse_args(argv)
    _install_signal_handlers()
    if args.list:
        return cmd_list(cfg)
    return watch(cfg)


if __name__ == "__main__":
    sys.exit(main())
