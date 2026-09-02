"""Configuration loading for the network-level study.

Mirrors miraca_uq.paths deliberately: overrides travel through environment
variables so they survive into MultiprocessingEvaluator workers, which
re-import this module fresh in each spawned process (Windows has no fork).

The one structural difference is `intermediate_dir`, which points at the
direct-damage repo rather than at this one - Stage 1 hazard intensities are
reused, never recomputed.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.network.yml"
DEFAULT_SCENARIO = "net_flood_absprot"


def set_country_override(country: str | None) -> None:
    if country:
        os.environ["MIRACA_NET_COUNTRY"] = country.upper()


def set_scenario_override(scenario: str | None) -> None:
    if scenario:
        os.environ["MIRACA_NET_SCENARIO"] = scenario.lower()


def set_archetype_override(archetype: str | None) -> None:
    if archetype:
        os.environ["MIRACA_NET_ARCHETYPE"] = archetype


def set_duration_override(mode: str | None) -> None:
    if mode:
        os.environ["MIRACA_NET_DURATION"] = mode


def set_hourly_override(hourly: bool) -> None:
    if hourly:
        os.environ["MIRACA_NET_HOURLY"] = "1"


def _resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_config(config_path: Path | None = None) -> dict:
    """Load config.network.yml, or whatever MIRACA_NET_CONFIG points at."""
    if config_path is None:
        env_cfg = os.environ.get("MIRACA_NET_CONFIG")
        config_path = Path(env_cfg) if env_cfg else CONFIG_PATH
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    env_country = os.environ.get("MIRACA_NET_COUNTRY")
    if env_country:
        cfg["country"] = env_country
    cfg["scenario"] = os.environ.get("MIRACA_NET_SCENARIO", DEFAULT_SCENARIO)
    # The timing archetype is a conditioning dimension chosen per run, and it
    # keys the per-process data cache, so it has to reach spawned workers the
    # same way the country does.
    env_arch = os.environ.get("MIRACA_NET_ARCHETYPE")
    if env_arch:
        cfg.setdefault("network", {})["timing_archetype"] = env_arch
    env_dur = os.environ.get("MIRACA_NET_DURATION")
    if env_dur:
        cfg.setdefault("network", {})["duration_sampling"] = env_dur
    env_hr = os.environ.get("MIRACA_NET_HOURLY")
    if env_hr:
        cfg.setdefault("network", {})["hourly_resolve"] = env_hr.lower() in ("1", "true", "yes")

    for key in ("results_dir", "cache_dir"):
        d = _resolve(cfg[key])
        d.mkdir(parents=True, exist_ok=True)
        cfg[key] = d
    for key in ("intermediate_dir", "energy_model_dir", "vulnerability_path",
                "fragility_path", "basin_data_path"):
        cfg[key] = _resolve(cfg[key])

    if not cfg["intermediate_dir"].is_dir():
        raise FileNotFoundError(
            f"intermediate_dir {cfg['intermediate_dir']} not found - Stage 1 output "
            "from the direct-damage study is required (it is not recomputed here)."
        )
    return cfg


def country_results_dir(cfg: dict, create: bool = True) -> Path:
    d = cfg["results_dir"] / cfg["country"]
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


def base_stem(cfg: dict) -> str:
    """Stage 1 filename stem in the direct-damage repo, e.g. 'LUX_power'."""
    return f"{cfg['country']}_{cfg['asset_type']}"


def result_stem(cfg: dict) -> str:
    return f"{cfg['country']}_powergrid_{cfg['scenario']}"


def grid_cache_path(cfg: dict) -> Path:
    """Stage 1b artefact: grid topology + injections, country-independent."""
    return cfg["cache_dir"] / "grid_cache.npz"


def hazard_cache_path(cfg: dict, hazard: str) -> Path:
    """Per-country, per-hazard component intensities (Stage 1 -> grid)."""
    return cfg["cache_dir"] / f"{cfg['country']}_powergrid_{hazard}.parquet"
