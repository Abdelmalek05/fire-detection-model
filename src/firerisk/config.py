# src/firerisk/config.py
import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv


@dataclass(frozen=True)
class Config:
    bbox: tuple
    resolution: float
    year_start: int
    year_end: int
    season_start: str
    season_end: str
    warmup_start: str
    data_dir: Path
    firms_source: str
    firms_types: list
    firms_confidence: list
    firms_day_range: int
    k_negatives: int
    temporal_buffer_days: int
    max_positives_per_cell_season: int
    random_seed: int
    universe_min_fire_days: int
    universe_min_years: int
    weather_chunk_size: int
    weather_sleep_seconds: float
    splits: dict
    map_key: str

    @property
    def years(self) -> list:
        return list(range(self.year_start, self.year_end + 1))


def load_config(path=None) -> Config:
    load_dotenv()
    path = Path(path) if path else Path("config/config.yaml")
    raw = yaml.safe_load(path.read_text())
    key = os.environ.get("FIRMS_MAP_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "FIRMS_MAP_KEY missing. Copy .env.example to .env and set it."
        )
    return Config(
        bbox=tuple(raw["bbox"]),
        resolution=float(raw["resolution"]),
        year_start=raw["years"][0],
        year_end=raw["years"][1],
        season_start=raw["season"]["start"],
        season_end=raw["season"]["end"],
        warmup_start=raw["warmup_start"],
        data_dir=Path(raw["data_dir"]),
        firms_source=raw["firms"]["source"],
        firms_types=raw["firms"]["types"],
        firms_confidence=raw["firms"]["confidence"],
        firms_day_range=raw["firms"]["day_range"],
        k_negatives=raw["sampling"]["k_negatives"],
        temporal_buffer_days=raw["sampling"]["temporal_buffer_days"],
        max_positives_per_cell_season=raw["sampling"]["max_positives_per_cell_season"],
        random_seed=raw["sampling"]["random_seed"],
        universe_min_fire_days=raw["universe"]["min_fire_days"],
        universe_min_years=raw["universe"]["min_distinct_years"],
        weather_chunk_size=raw["weather"]["chunk_size"],
        weather_sleep_seconds=raw["weather"]["sleep_seconds"],
        splits=raw["splits"],
        map_key=key,
    )
