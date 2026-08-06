"""
config.py
=========

Central configuration for the Swing Scanner project.

Responsibilities
----------------
- Load environment variables from .env
- Define strongly typed application settings
- Expose a single `settings` object
- Provide project paths and runtime configuration

Usage
-----
from config import settings

settings.paths.raw
settings.storage.duckdb
settings.scanner.top_stocks
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# =============================================================================
# Project Paths
# =============================================================================


PROJECT_ROOT = Path(__file__).resolve().parent


class PathSettings:
    """Filesystem paths."""

    root = PROJECT_ROOT

    data = root / "data"

    raw = data / "raw"

    cache = data / "cache"

    exports = data / "exports"

    reports = data / "reports"


# =============================================================================
# Application
# =============================================================================


class AppSettings(BaseSettings):
    """Application configuration."""

    name: str = Field(default="Swing Scanner")
    environment: str = Field(default="development")
    version: str = Field(default="1.0.0")
    timezone: str = Field(default="Asia/Kolkata")


# =============================================================================
# Market Data
# =============================================================================


class MarketSettings(BaseSettings):
    """Market data providers."""

    yahoo_enabled: bool = True
    nse_enabled: bool = True

    request_timeout: int = 30

    retries: int = 3

    lookback_days: int = 3650


# =============================================================================
# Screener
# =============================================================================


class ScreenerSettings(BaseSettings):
    """Screening configuration."""

    min_market_cap: float = 0.0

    min_volume: int = 100000

    enable_governance_filter: bool = True

    enable_news_filter: bool = True


# =============================================================================
# Scanner
# =============================================================================


class ScannerSettings(BaseSettings):
    """Scanner configuration."""

    top_stocks: int = 20

    min_score: float = 0.0


# =============================================================================
# Backtest
# =============================================================================


class BacktestSettings(BaseSettings):
    """Backtesting configuration."""

    enabled: bool = True

    years: int = 10

    initial_capital: float = 1_000_000.0


# =============================================================================
# Storage
# =============================================================================


class StorageSettings(BaseSettings):
    """Storage configuration."""

    duckdb: Path = PathSettings.exports / "swing_scanner.duckdb"

    parquet: Path = PathSettings.cache

    excel: Path = PathSettings.reports


# =============================================================================
# Logging
# =============================================================================


class LoggingSettings(BaseSettings):
    """Logging configuration."""

    level: str = "INFO"

    file: Path = PathSettings.reports / "scanner.log"


# =============================================================================
# GitHub
# =============================================================================


class GitHubSettings(BaseSettings):
    """GitHub Actions configuration."""

    enabled: bool = True

    artifacts: bool = True


# =============================================================================
# Root Settings
# =============================================================================


class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app: AppSettings = AppSettings()

    paths: PathSettings = PathSettings()

    market: MarketSettings = MarketSettings()

    screener: ScreenerSettings = ScreenerSettings()

    scanner: ScannerSettings = ScannerSettings()

    backtest: BacktestSettings = BacktestSettings()

    storage: StorageSettings = StorageSettings()

    logging: LoggingSettings = LoggingSettings()

    github: GitHubSettings = GitHubSettings()


settings = Settings()