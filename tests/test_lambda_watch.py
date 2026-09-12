"""Tests for tools/lambda_watch.py.

Covers the pure selection/classification logic and the launch payload, plus one
end-to-end pass of the poll loop against a stubbed client. Nothing here touches
the network.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

_SRC = pathlib.Path(__file__).resolve().parents[1] / "lambda_watch.py"
_spec = importlib.util.spec_from_file_location("lambda_watch", _SRC)
lw = importlib.util.module_from_spec(_spec)
sys.modules["lambda_watch"] = lw
_spec.loader.exec_module(lw)


def itype(name, gpu_desc, gpus, price, regions=()):
    return {
        name: {
            "instance_type": {
                "name": name,
                "gpu_description": gpu_desc,
                "price_cents_per_hour": price,
                "specs": {"gpus": gpus},
            },
            "regions_with_capacity_available": [{"name": r} for r in regions],
        }
    }


# Shaped like the live /instance-types payload, trimmed to what we filter on.
PAYLOAD = {
    **itype("gpu_1x_b200_sxm6", "B200 (180 GB SXM6)", 1, 699),
    **itype("gpu_8x_b200_sxm6", "B200 (180 GB SXM6)", 8, 5352, ["us-west-1"]),
    **itype("gpu_1x_h100_pcie", "H100 (80 GB PCIe)", 1, 329, ["us-west-3"]),
    **itype("gpu_1x_a10", "A10 (24 GB PCIe)", 1, 129, ["us-east-1"]),
    **itype("cpu_4x_general", "N/A", 0, 20, ["us-west-1"]),
}


def with_capacity(regions):
    """PAYLOAD, but the 1x B200 has capacity in `regions`."""
    out = {k: dict(v) for k, v in PAYLOAD.items()}
    out["gpu_1x_b200_sxm6"]["regions_with_capacity_available"] = [{"name": r} for r in regions]
    return out


DEFAULTS = dict(gpu_count=1, gpu_pattern=lw.BLACKWELL_GPU_PATTERN, explicit_types=None)
FIND_DEFAULTS = dict(**DEFAULTS, allowed_regions=None, max_price_cents=None)


class TestTargetMatching:
    def test_only_the_1x_blackwell_is_a_target(self):
        assert lw.matching_target_types(PAYLOAD, **DEFAULTS) == ["gpu_1x_b200_sxm6"]

    @pytest.mark.parametrize(
        "desc",
        [
            "B200 (180 GB SXM6)",
            "GB200 (192 GB)",
            "B300 (288 GB)",
            "RTX PRO 6000 Blackwell (96 GB)",
            "RTX 6000 Blackwell Server Edition",
        ],
    )
    def test_future_blackwell_names_match(self, desc):
        """New Lambda SKUs in the family should be picked up without a code change."""
        assert lw.matches_target("gpu_1x_x", desc, 1, **DEFAULTS)

    @pytest.mark.parametrize(
        "desc", ["H100 (80 GB SXM5)", "A100 (40 GB PCIe)", "GH200 (96 GB)", "RTX 6000 (24 GB)", "N/A"]
    )
    def test_non_blackwell_names_do_not_match(self, desc):
        """Notably RTX 6000 (Turing) and GH200 (Hopper) must not be mistaken for Blackwell."""
        assert not lw.matches_target("gpu_1x_x", desc, 1, **DEFAULTS)

    def test_gpu_count_is_enforced(self):
        assert not lw.matches_target("gpu_8x_b200_sxm6", "B200 (180 GB SXM6)", 8, **DEFAULTS)

    def test_explicit_type_overrides_family_and_count(self):
        kw = dict(DEFAULTS, explicit_types=["gpu_8x_h100_sxm5"])
        assert lw.matches_target("gpu_8x_h100_sxm5", "H100 (80 GB SXM5)", 8, **kw)
        assert not lw.matches_target("gpu_1x_b200_sxm6", "B200 (180 GB SXM6)", 1, **kw)


class TestFindCandidates:
    def test_no_capacity_yields_nothing(self):
        assert lw.find_candidates(PAYLOAD, **FIND_DEFAULTS) == []

    def test_capacity_is_found(self):
        got = lw.find_candidates(with_capacity(["us-west-1"]), **FIND_DEFAULTS)
        assert len(got) == 1
        assert got[0].instance_type == "gpu_1x_b200_sxm6"
        assert got[0].region == "us-west-1"
        assert got[0].price_str == "$6.99/hr"

    def test_one_candidate_per_region(self):
        got = lw.find_candidates(with_capacity(["us-west-1", "us-east-1"]), **FIND_DEFAULTS)
        assert {c.region for c in got} == {"us-west-1", "us-east-1"}

    def test_region_filter_excludes_others(self):
        kw = dict(FIND_DEFAULTS, allowed_regions=["us-east-1"])
        got = lw.find_candidates(with_capacity(["us-west-1", "us-east-1"]), **kw)
        assert [c.region for c in got] == ["us-east-1"]

    def test_region_order_is_a_preference_order(self):
        kw = dict(FIND_DEFAULTS, allowed_regions=["us-east-1", "us-west-1"])
        got = lw.find_candidates(with_capacity(["us-west-1", "us-east-1"]), **kw)
        assert [c.region for c in got] == ["us-east-1", "us-west-1"]

    def test_price_ceiling_blocks_an_expensive_launch(self):
        kw = dict(FIND_DEFAULTS, max_price_cents=500)
        assert lw.find_candidates(with_capacity(["us-west-1"]), **kw) == []

    def test_price_ceiling_at_the_exact_price_allows_it(self):
        kw = dict(FIND_DEFAULTS, max_price_cents=699)
        assert len(lw.find_candidates(with_capacity(["us-west-1"]), **kw)) == 1

    def test_cheapest_first_within_a_region(self):
        kw = dict(FIND_DEFAULTS, gpu_pattern=r".")  # match every 1-GPU type
        got = lw.find_candidates(with_capacity(["us-east-1"]), **kw)
        assert [c.price_cents_per_hour for c in got] == sorted(
            c.price_cents_per_hour for c in got
        )

    def test_empty_and_malformed_payloads_are_survivable(self):
        assert lw.find_candidates({}, **FIND_DEFAULTS) == []
        assert lw.find_candidates({"x": {}}, **FIND_DEFAULTS) == []


class TestExistingInstances:
    TARGETS = ["gpu_1x_b200_sxm6"]

    def inst(self, tname, status):
        return {"id": "abc", "status": status, "instance_type": {"name": tname}}

    @pytest.mark.parametrize("status", ["active", "booting", "unhealthy"])
    def test_live_target_instance_is_reported(self, status):
        got = lw.existing_target_instances([self.inst("gpu_1x_b200_sxm6", status)], self.TARGETS)
        assert len(got) == 1

    @pytest.mark.parametrize("status", ["terminated", "terminating"])
    def test_dead_instance_is_ignored(self, status):
        assert lw.existing_target_instances([self.inst("gpu_1x_b200_sxm6", status)], self.TARGETS) == []

    def test_other_instance_types_are_ignored(self):
        """The user's running H100 must not suppress a B200 hunt."""
        assert lw.existing_target_instances([self.inst("gpu_1x_h100_pcie", "active")], self.TARGETS) == []

    def test_empty_list(self):
        assert lw.existing_target_instances([], self.TARGETS) == []


class TestErrorClassification:
    def test_insufficient_capacity_is_retryable(self):
        err = lw.ApiError(400, lw.INSUFFICIENT_CAPACITY, "no capacity")
        assert err.retryable

    @pytest.mark.parametrize(
        "code",
        [
            "global/quota-exceeded",
            "global/invalid-parameters",
            "global/account-inactive",
            "global/invalid-address",
            "global/invalid-api-key",
            "global/forbidden",
            "global/not-found",
            "global/object-does-not-exist",
            "instance-operations/launch/file-system-in-wrong-region",
        ],
    )
    def test_account_level_errors_are_fatal(self, code):
        assert not lw.ApiError(400, code, "nope").retryable

    def test_internal_error_is_retryable(self):
        assert lw.ApiError(500, "global/internal-error", "oops").retryable

    def test_server_errors_and_429_are_retryable(self):
        assert lw.ApiError(503, "", "unavailable").retryable
        assert lw.ApiError(429, "", "slow down").retryable

    def test_a_real_invalid_key_401_is_fatal(self):
        """Exactly what the live API returns for a bad key."""
        err = lw.ApiError(401, "global/invalid-api-key", "API key was invalid, expired, or deleted.")
        assert not err.retryable

    def test_unknown_4xx_is_fatal(self):
        """Better to stop than to hammer a request that can never succeed."""
        assert not lw.ApiError(418, "weird/thing", "teapot").retryable

    def test_str_includes_the_useful_parts(self):
        s = str(lw.ApiError(400, "a/b", "msg", suggestion="do x", request_id="r1"))
        assert "HTTP 400" in s and "a/b" in s and "msg" in s and "do x" in s and "r1" in s


class TestLaunchPayload:
    def cand(self):
        return lw.Candidate("gpu_1x_b200_sxm6", "us-west-1", "B200 (180 GB SXM6)", 1, 699)

    def test_minimal_payload_has_exactly_the_required_fields(self):
        cfg = lw.Config(api_key="k", ssh_keys=["mykey"])
        assert lw.build_launch_payload(cfg, self.cand()) == {
            "region_name": "us-west-1",
            "instance_type_name": "gpu_1x_b200_sxm6",
            "ssh_key_names": ["mykey"],
        }

    def test_optional_fields_are_included_when_set(self):
        cfg = lw.Config(
            api_key="k",
            ssh_keys=["mykey"],
            name="b200",
            image_family="lambda-stack-24-04",
            file_systems=["fs1"],
        )
        p = lw.build_launch_payload(cfg, self.cand())
        assert p["name"] == "b200"
        assert p["image"] == {"family": "lambda-stack-24-04"}
        assert p["file_system_names"] == ["fs1"]

    def test_user_data_is_read_from_disk(self, tmp_path):
        f = tmp_path / "init.sh"
        f.write_text("#!/bin/bash\necho hi\n")
        cfg = lw.Config(api_key="k", ssh_keys=["mykey"], user_data_file=str(f))
        assert lw.build_launch_payload(cfg, self.cand())["user_data"] == "#!/bin/bash\necho hi\n"


class TestDurationParsing:
    @pytest.mark.parametrize(
        "text,secs",
        [("30", 30), ("30s", 30), ("15m", 900), ("6h", 21600), ("1d", 86400), ("1.5h", 5400), (" 45S ", 45)],
    )
    def test_valid(self, text, secs):
        assert lw.parse_duration(text) == secs

    @pytest.mark.parametrize("text", ["", "soon", "5y", "-30s", "m"])
    def test_invalid(self, text):
        with pytest.raises(SystemExit):
            lw.parse_duration(text)


class TestApiKeyFile:
    def test_picks_the_secret_line_out_of_a_labelled_file(self):
        text = "Lambda API key for mosaicist-testing\n\nsecret_abc_123.xyz\n"
        assert lw._first_key_line(text) == "secret_abc_123.xyz"

    def test_bare_key_file(self):
        assert lw._first_key_line("secret_abc_123.xyz\n") == "secret_abc_123.xyz"

    def test_falls_back_to_the_first_line(self):
        assert lw._first_key_line("someothertoken\n") == "someothertoken"

    def test_empty_file_is_an_error(self):
        with pytest.raises(SystemExit):
            lw._first_key_line("   \n\n")

    def test_env_var_is_used_when_no_flag(self, monkeypatch):
        monkeypatch.setenv("LAMBDA_API_KEY", "secret_env")
        assert lw.read_api_key(None, None) == "secret_env"

    def test_explicit_flag_beats_env(self, monkeypatch):
        monkeypatch.setenv("LAMBDA_API_KEY", "secret_env")
        assert lw.read_api_key("secret_flag", None) == "secret_flag"

    def test_file_beats_env(self, monkeypatch, tmp_path):
        monkeypatch.setenv("LAMBDA_API_KEY", "secret_env")
        f = tmp_path / "k.txt"
        f.write_text("secret_file\n")
        assert lw.read_api_key(None, str(f)) == "secret_file"

    def test_missing_everywhere_is_an_error(self, monkeypatch, tmp_path):
        monkeypatch.delenv("LAMBDA_API_KEY", raising=False)
        monkeypatch.setattr(lw.os.path, "expanduser", lambda p: str(tmp_path / "nope"))
        with pytest.raises(SystemExit):
            lw.read_api_key(None, None)


class TestCli:
    def test_ssh_key_required_to_launch(self, monkeypatch):
        monkeypatch.setenv("LAMBDA_API_KEY", "secret_env")
        with pytest.raises(SystemExit):
            lw.parse_args([])

    def test_dry_run_does_not_require_an_ssh_key(self, monkeypatch):
        monkeypatch.setenv("LAMBDA_API_KEY", "secret_env")
        cfg, _ = lw.parse_args(["--dry-run"])
        assert cfg.dry_run and cfg.ssh_keys == []

    def test_flags_map_onto_config(self, monkeypatch):
        monkeypatch.setenv("LAMBDA_API_KEY", "secret_env")
        cfg, _ = lw.parse_args(
            [
                "--ssh-key", "k1",
                "--region", "us-west-1",
                "--region", "us-east-1",
                "--max-price", "8.00",
                "--interval", "15s",
                "--timeout", "2h",
                "--no-wait",
            ]
        )
        assert cfg.ssh_keys == ["k1"]
        assert cfg.regions == ["us-west-1", "us-east-1"]
        assert cfg.max_price_cents == 800
        assert cfg.interval == 15
        assert cfg.timeout_s == 7200
        assert cfg.wait_for_active is False


class FakeClient:
    """Stand-in for LambdaClient: serves scripted /instance-types payloads and
    records launch attempts."""

    def __init__(self, payloads, *, launch_results=None, keys=("mykey",), instances=()):
        self._payloads = list(payloads)
        self._launch_results = list(launch_results or [])
        self._keys = keys
        self._instances = list(instances)
        self.launches = []

    def ssh_keys(self):
        return [{"name": k} for k in self._keys]

    def instances(self):
        return self._instances

    def instance(self, instance_id):
        return {"id": instance_id, "status": "active", "ip": "1.2.3.4"}

    def instance_types(self):
        return self._payloads.pop(0) if self._payloads else PAYLOAD

    def launch(self, payload):
        self.launches.append(payload)
        result = self._launch_results.pop(0) if self._launch_results else ["id-1"]
        if isinstance(result, Exception):
            raise result
        return result


@pytest.fixture
def fast(monkeypatch):
    """Neuter sleeps and notifications so loop tests run instantly and silently."""
    monkeypatch.setattr(lw.time, "sleep", lambda *_: None)
    monkeypatch.setattr(lw, "notify", lambda *a, **k: None)
    monkeypatch.setattr(lw, "MIN_LAUNCH_INTERVAL_S", 0)


def run_watch(monkeypatch, client, cfg):
    monkeypatch.setattr(lw, "LambdaClient", lambda *a, **k: client)
    monkeypatch.setattr(lw, "_stop", False)
    return lw.watch(cfg)


class TestWatchLoop:
    def cfg(self, **kw):
        return lw.Config(api_key="k", ssh_keys=["mykey"], interval=0, jitter=0, **kw)

    def test_launches_when_capacity_appears_after_a_miss(self, monkeypatch, fast):
        client = FakeClient([PAYLOAD, with_capacity(["us-west-1"])])
        assert run_watch(monkeypatch, client, self.cfg()) == 0
        assert len(client.launches) == 1
        assert client.launches[0]["instance_type_name"] == "gpu_1x_b200_sxm6"
        assert client.launches[0]["region_name"] == "us-west-1"

    def test_dry_run_never_launches(self, monkeypatch, fast):
        client = FakeClient([with_capacity(["us-west-1"])])
        assert run_watch(monkeypatch, client, self.cfg(dry_run=True, once=True)) == 0
        assert client.launches == []

    def test_losing_the_race_keeps_hunting_then_wins(self, monkeypatch, fast):
        """insufficient-capacity is the normal outcome of a lost race, not a stop."""
        client = FakeClient(
            [with_capacity(["us-west-1"]), with_capacity(["us-west-1"])],
            launch_results=[lw.ApiError(400, lw.INSUFFICIENT_CAPACITY, "no"), ["id-2"]],
        )
        assert run_watch(monkeypatch, client, self.cfg()) == 0
        assert len(client.launches) == 2

    def test_quota_exceeded_stops_immediately(self, monkeypatch, fast):
        client = FakeClient(
            [with_capacity(["us-west-1"])],
            launch_results=[lw.ApiError(400, "global/quota-exceeded", "quota")],
        )
        assert run_watch(monkeypatch, client, self.cfg()) == 2
        assert len(client.launches) == 1

    def test_existing_instance_blocks_a_duplicate_launch(self, monkeypatch, fast):
        client = FakeClient(
            [with_capacity(["us-west-1"])],
            instances=[
                {
                    "id": "x",
                    "status": "active",
                    "ip": "9.9.9.9",
                    "instance_type": {"name": "gpu_1x_b200_sxm6"},
                    "region": {"name": "us-west-1"},
                }
            ],
        )
        assert run_watch(monkeypatch, client, self.cfg()) == 0
        assert client.launches == []

    def test_allow_duplicate_overrides_that(self, monkeypatch, fast):
        client = FakeClient(
            [with_capacity(["us-west-1"])],
            instances=[
                {
                    "id": "x",
                    "status": "active",
                    "instance_type": {"name": "gpu_1x_b200_sxm6"},
                    "region": {"name": "us-west-1"},
                }
            ],
        )
        assert run_watch(monkeypatch, client, self.cfg(allow_duplicate=True)) == 0
        assert len(client.launches) == 1

    def test_once_with_no_capacity_exits_1(self, monkeypatch, fast):
        client = FakeClient([PAYLOAD])
        assert run_watch(monkeypatch, client, self.cfg(once=True)) == 1
        assert client.launches == []

    def test_unknown_ssh_key_fails_preflight(self, monkeypatch, fast):
        client = FakeClient([PAYLOAD], keys=("someoneelses",))
        assert run_watch(monkeypatch, client, self.cfg()) == 2
        assert client.launches == []

    def test_filters_matching_nothing_fail_preflight(self, monkeypatch, fast):
        client = FakeClient([PAYLOAD])
        cfg = self.cfg(instance_types=["gpu_1x_nonexistent"])
        assert run_watch(monkeypatch, client, cfg) == 2

    def test_timeout_returns_3(self, monkeypatch, fast):
        """With sleeps neutered, advance a fake clock so the deadline is reachable."""
        client = FakeClient([PAYLOAD] * 50)
        clock = iter(range(0, 10_000, 10))
        monkeypatch.setattr(lw.time, "monotonic", lambda: float(next(clock)))
        assert run_watch(monkeypatch, client, self.cfg(timeout_s=60)) == 3
        assert client.launches == []

    def test_a_second_region_is_tried_after_the_first_loses(self, monkeypatch, fast):
        client = FakeClient(
            [with_capacity(["us-west-1", "us-east-1"])],
            launch_results=[lw.ApiError(400, lw.INSUFFICIENT_CAPACITY, "no"), ["id-3"]],
        )
        cfg = self.cfg(regions=["us-west-1", "us-east-1"])
        assert run_watch(monkeypatch, client, cfg) == 0
        assert [p["region_name"] for p in client.launches] == ["us-west-1", "us-east-1"]
