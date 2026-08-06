"""
engine/data.py
==============

Institutional Market Data Engine.

Responsibilities
----------------
* Load trading universe
* Download market data
* Download benchmark data
* Download sector indices
* Normalize raw market data
* Validate data quality
* Enrich market data
* Persist datasets
* Maintain cached market data

This module is the ONLY component responsible for
market data acquisition and enrichment.
"""

from __future__ import annotations

###############################################################################
# Standard Library
###############################################################################
import logging
import time
from datetime import datetime
from typing import Any

###############################################################################
# Third Party
###############################################################################
import pandas as pd
import yfinance as yf

###############################################################################
# Local Imports
###############################################################################
from config import settings
from engine.governance_fetcher import (
    refresh_governance_overrides,
)
from engine.storage import StorageManager
from engine.universe_loader import load_full_universe
from engine.news_sentiment import fetch_news_score

###############################################################################
# Logger
###############################################################################

logger = logging.getLogger(__name__)

###############################################################################
# Constants
###############################################################################

DEFAULT_PERIOD = "10y"

DEFAULT_INTERVAL = "1d"

DEFAULT_BATCH_SIZE = 25

DEFAULT_RETRIES = 3

DEFAULT_SLEEP_SECONDS = 2

DEFAULT_BENCHMARK = "^NSEI"

###############################################################################
# Market Data Engine
###############################################################################


class MarketDataEngine:
    """
    Central market data engine.

    Pipeline
    --------

        Universe

            ↓

        Batch Download

            ↓

        Benchmark

            ↓

        Normalization

            ↓

        Validation

            ↓

        Enrichment

            ↓

        Analytics

            ↓

        Storage
    """

    ###########################################################################
    # Initialization
    ###########################################################################

    def __init__(
        self,
    ) -> None:
        """
        Initialize MarketDataEngine.
        """

        self.settings = settings

        self.storage = StorageManager()

        bundle = load_full_universe()

        self.bundle = bundle

        self.buckets = bundle["buckets"]

        self.sector_map = bundle["sector_map"]

        self.meta = bundle["meta"]

        self.market_data = pd.DataFrame()

        self.benchmark_data = pd.DataFrame()

        self.market_breadth: dict[str, Any] = {}

        self.last_refresh: datetime | None = None

        logger.info(
            "MarketDataEngine initialized."
        )

    ###########################################################################
    # Universe
    ###########################################################################

    @property
    def universe(
        self,
    ) -> list[str]:
        """
        Return master universe.
        """

        return self.buckets.get(
            "AllNSE",
            [],
        )[:5]


    def get_universe(
        self,
        bucket: str = "AllNSE",
    ) -> list[str]:
        """
        Return requested universe bucket.
        """

        return self.buckets.get(
            bucket,
            [],
        )

    def yahoo_symbols(
        self,
        symbols: list[str] | None = None,
    ) -> list[str]:
        """
        Convert NSE symbols to Yahoo Finance format.
        """

        symbols = symbols or self.universe

        return [

            symbol
            if symbol.startswith("^")
            or symbol.endswith(".NS")
            else f"{symbol}.NS"

            for symbol in symbols

        ]

    def available_universes(
        self,
    ) -> list[str]:
        """
        Return available universe names.
        """

        return sorted(
            self.buckets.keys(),
        )

    ###########################################################################
    # Download Helpers
    ###########################################################################

    def _batched_symbols(
        self,
        symbols: list[str],
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> list[list[str]]:
        """
        Split symbols into batches.
        """

        return [

            symbols[index:index + batch_size]

            for index in range(
                0,
                len(symbols),
                batch_size,
            )
        ]


    def _download_batch(
        self,
        symbols: list[str],
        *,
        period: str,
        interval: str,
        auto_adjust: bool,
        progress: bool,
    ) -> pd.DataFrame:
        """
        Download a single symbol batch.
        """

        yahoo_symbols = self.yahoo_symbols(
            symbols,
        )

        return yf.download(

            tickers=yahoo_symbols,
            period=period,
            interval=interval,
            auto_adjust=auto_adjust,
            group_by="ticker",
            threads=True,
            progress=progress,

        )


    def _retry_download(
        self,
        symbols: list[str],
        *,
        period: str,
        interval: str,
        auto_adjust: bool,
        progress: bool,
    ) -> pd.DataFrame:
        """
        Download with retry logic.
        """

        last_exception: Exception | None = None

        for attempt in range(

            DEFAULT_RETRIES,

        ):

            try:

                logger.info(

                    "Downloading batch %d/%d.",

                    attempt + 1,

                    DEFAULT_RETRIES,

                )

                return self._download_batch(

                    symbols,

                    period=period,

                    interval=interval,

                    auto_adjust=auto_adjust,

                    progress=progress,

                )

            except Exception as exc:

                last_exception = exc

                logger.warning(

                    "Download failed (%d/%d): %s",

                    attempt + 1,

                    DEFAULT_RETRIES,

                    exc,

                )

                time.sleep(
                    DEFAULT_SLEEP_SECONDS
                )

        raise RuntimeError(
            "Market data download failed."
        ) from last_exception


    ###########################################################################
    # Download Engine
    ###########################################################################

    def download_market_data(
        self,
        *,
        period: str = DEFAULT_PERIOD,
        interval: str = DEFAULT_INTERVAL,
        auto_adjust: bool = False,
        progress: bool = False,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> pd.DataFrame:
        """
        Download market data in batches.
        """

        logger.info(
            "Market data download started. Symbols=%d",
            len(self.universe),
        )

        if not self.universe:

            raise ValueError(
                "Universe is empty."
            )

        frames: list[pd.DataFrame] = []

        logger.info(
            "Preparing download batches..."
        )

        batches = self._batched_symbols(

            self.universe,
            batch_size,

        )

        total_batches = len(batches)

        logger.info(
            "Total market data batches: %d",
            total_batches,
        )

        for index, batch in enumerate(

            batches,
            start=1,

        ):

            logger.info(

                "Downloading market data batch %d/%d (%d symbols)",
                index,
                total_batches,
                len(batch),

            )


            downloaded = self._retry_download(

                batch,
                period=period,
                interval=interval,
                auto_adjust=auto_adjust,
                progress=progress,

            )

            frames.append(
                downloaded
            )

        dataframe = pd.concat(

            frames,
            axis=1,

        )

        dataframe = self.normalize_market_data(
            dataframe,
        )

        dataframe = self.validate_market_data(
            dataframe,
        )

        self.market_data = dataframe

        self.last_refresh = datetime.utcnow()

        logger.info(
            "Downloaded %d rows.",
            len(dataframe),
        )

        return dataframe

    ###########################################################################
    # Benchmark
    ###########################################################################

    def download_benchmark(
        self,
        symbol: str = DEFAULT_BENCHMARK,
        *,
        period: str = DEFAULT_PERIOD,
        interval: str = DEFAULT_INTERVAL,
    ) -> pd.DataFrame:
        """
        Download benchmark index.
        """

        logger.info(
            "Downloading benchmark %s.",
            symbol,
        )

        benchmark = yf.download(

            tickers=symbol,

            period=period,

            interval=interval,

            auto_adjust=False,

            progress=False,

        )

        benchmark = benchmark.reset_index()

        benchmark["Benchmark"] = symbol

        benchmark = benchmark.rename(

            columns={

                "Close": "Benchmark_Close",

            }

        )

        self.benchmark_data = benchmark

        return benchmark

    ###########################################################################
    # Sector Indices
    ###########################################################################

    def download_sector_indices(
        self,
        sectors: list[str],
        *,
        period: str = DEFAULT_PERIOD,
        interval: str = DEFAULT_INTERVAL,
    ) -> dict[str, pd.DataFrame]:
        """
        Download sector index history.
        """

        logger.info(
            "Downloading sector indices."
        )

        output: dict[str, pd.DataFrame] = {}

        for sector in sectors:

            try:

                output[sector] = yf.download(

                    sector,

                    period=period,

                    interval=interval,

                    progress=False,

                )

            except Exception as exc:

                logger.warning(

                    "Sector download failed: %s",

                    exc,

                )

        return output

    ###########################################################################
    # Normalization
    ###########################################################################

    def normalize_market_data(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Normalize Yahoo Finance output.

        Converts MultiIndex columns into a
        long-format DataFrame.
        """

        if dataframe.empty:

            return dataframe

        dataframe = dataframe.copy()

        if isinstance(
            dataframe.columns,
            pd.MultiIndex,
        ):

            dataframe = (

                dataframe
                .stack(level=0)
                .rename_axis(

                    [
                        "Date",
                        "Symbol",
                    ]
                )

                .reset_index()
            )

        dataframe.columns = [

            str(column).strip()
            for column in dataframe.columns
        ]

        dataframe["Symbol"] = (

            dataframe["Symbol"]
            .str.replace(
                ".NS",
                "",
                regex=False,
            )
        )

        dataframe = dataframe.sort_values(

            [
                "Symbol",
                "Date",
            ]
        )

        dataframe = dataframe.reset_index(
            drop=True,
        )

        return dataframe

    ###########################################################################
    # Validation
    ###########################################################################

    def validate_market_data(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Validate downloaded data.
        """

        logger.info(
            "Validating market data."
        )

        if dataframe.empty:

            raise ValueError(
                "Downloaded dataframe is empty."
            )

        dataframe = dataframe.copy()

        dataframe = dataframe.drop_duplicates(

            subset=[

                "Symbol",

                "Date",

            ]

        )

        dataframe = dataframe.dropna(

            subset=[

                "Open",

                "High",

                "Low",

                "Close",

                "Volume",

            ],

            how="any",

        )

        dataframe["Volume"] = (

            dataframe["Volume"]

            .clip(lower=0)

        )

        dataframe = dataframe.sort_values(

            [

                "Symbol",

                "Date",

            ]

        )

        dataframe = dataframe.reset_index(

            drop=True,

        )

        logger.info(
            "Validation completed."
        )

        return dataframe

    ###########################################################################
    # Corporate Actions
    ###########################################################################

    def merge_corporate_actions(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Merge corporate actions.
        """

        logger.info(
            "Merging corporate actions."
        )

        dataframe = dataframe.copy()

        defaults = {

            "BONUS": False,

            "SPLIT": False,

            "DIVIDEND": 0.0,

            "RIGHTS": False,

        }

        try:

            from engine.corporate_actions import (
                CorporateActionsEngine,
            )

            actions = CorporateActionsEngine()

            action_df = actions.fetch_latest()

            required = {"Symbol"}

            if required.issubset(action_df.columns):

                dataframe = dataframe.merge(

                    action_df,

                    how="left",

                    on="Symbol",

                )

        except Exception as exc:

            logger.warning(
                "Corporate actions unavailable: %s",
                exc,
            )

        for column, value in defaults.items():

            if column not in dataframe.columns:

                dataframe[column] = value

            else:

                dataframe[column] = dataframe[
                    column
                ].fillna(value)

        return dataframe


    ###########################################################################
    # News Sentiment
    ###########################################################################

    def merge_news_sentiment(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Merge latest news sentiment.
        """

        logger.info(
            "Merging news sentiment."
        )

        dataframe = dataframe.copy()

        try:

            sentiment = pd.DataFrame(
                columns=[
                    "Symbol",
                    "NEWS_SCORE",
                ]
            )

            for symbol in dataframe["Symbol"].unique():

                result = fetch_news_score(
                    f"{symbol}.NS",
                )

                sentiment = pd.concat(

                    [

                        sentiment,

                        pd.DataFrame(
                            [
                                {
                                    "Symbol": symbol,
                                    "NEWS_SCORE": result["score"],
                                }
                            ]
                        ),

                    ],

                    ignore_index=True,

                )

            if {

                "Symbol",

                "NEWS_SCORE",

            }.issubset(sentiment.columns):

                dataframe = dataframe.merge(

                    sentiment,

                    how="left",

                    on="Symbol",

                )

        except Exception as exc:

            logger.warning(
                "News sentiment unavailable: %s",
                exc,
            )

        if "NEWS_SCORE" not in dataframe.columns:

            dataframe["NEWS_SCORE"] = 0.0

        else:

            dataframe["NEWS_SCORE"] = (
                dataframe["NEWS_SCORE"]
                .fillna(0.0)
            )

        return dataframe


    ###########################################################################
    # Governance
    ###########################################################################

    def merge_governance(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Merge governance scores.
        """

        logger.info(
            "Merging governance."
        )

        dataframe = dataframe.copy()

        try:


            governance = refresh_governance_overrides(
                dataframe["Symbol"].tolist()
            )

            if {
                "Symbol",
                "GOVERNANCE_SCORE",
            }.issubset(governance.columns):

                dataframe = dataframe.merge(
                    governance,
                    how="left",
                    on="Symbol",
                )

        except Exception as exc:

            logger.warning(
                "Governance unavailable: %s",
                exc,
            )

        if "GOVERNANCE_SCORE" not in dataframe.columns:

            dataframe["GOVERNANCE_SCORE"] = 100.0

        else:

            dataframe["GOVERNANCE_SCORE"] = (
                dataframe["GOVERNANCE_SCORE"]
                .fillna(100.0)
            )

        return dataframe


    ###########################################################################
    # NSE Events
    ###########################################################################

    def merge_nse_events(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Merge NSE corporate events.
        """

        logger.info(
            "Merging NSE events."
        )

        dataframe = dataframe.copy()

        try:

            from engine.nse_events import (
                NSEEventsEngine,
            )

            events = (

                NSEEventsEngine()

                .fetch_latest()

            )

            if {

                "Symbol",

                "EVENT_COUNT",

            }.issubset(events.columns):

                dataframe = dataframe.merge(

                    events,

                    how="left",

                    on="Symbol",

                )

        except Exception as exc:

            logger.warning(
                "NSE events unavailable: %s",
                exc,
            )

        dataframe["EVENT_COUNT"] = (

            dataframe.get(
                "EVENT_COUNT",
                0,
            )

            .fillna(0)

        )

        return dataframe


    ###########################################################################
    # Sector Mapping
    ###########################################################################

    def map_sectors(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Map sector metadata.
        """

        logger.info(
            "Mapping sectors."
        )

        dataframe = dataframe.copy()

        dataframe["Sector"] = (

            dataframe["Symbol"]

            .map(self.sector_map)

            .fillna("Unknown")

        )

        return dataframe


    ###########################################################################
    # Enrichment Pipeline
    ###########################################################################

    def enrich_market_data(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Execute market enrichment pipeline.
        """

        logger.info(
            "Starting enrichment."
        )

        dataframe = self.merge_corporate_actions(
            dataframe,
        )

        dataframe = self.merge_news_sentiment(
            dataframe,
        )

        dataframe = self.merge_governance(
            dataframe,
        )

        dataframe = self.merge_nse_events(
            dataframe,
        )

        dataframe = self.map_sectors(
            dataframe,
        )

        logger.info(
            "Market enrichment completed."
        )

        return dataframe

    ###########################################################################
    # Relative Strength
    ###########################################################################

    def calculate_relative_strength(
        self,
        dataframe: pd.DataFrame,
        benchmark: pd.DataFrame,
        *,
        lookback: int = 50,
    ) -> pd.DataFrame:
        """
        Calculate rolling relative strength.
        """

        logger.info(
            "Calculating relative strength."
        )

        dataframe = dataframe.copy()

        benchmark = benchmark.copy()

        benchmark = benchmark.rename(
            columns={
                "Benchmark_Close": "BENCHMARK_CLOSE",
            }
        )

        dataframe = dataframe.merge(

            benchmark[
                [
                    "Date",
                    "BENCHMARK_CLOSE",
                ]
            ],

            on="Date",

            how="left",

        )

        dataframe["STOCK_RETURN"] = (

            dataframe.groupby("Symbol")["Close"]

            .pct_change(lookback)

        )

        dataframe["BENCHMARK_RETURN"] = (

            dataframe["BENCHMARK_CLOSE"]

            .pct_change(lookback)

        )

        dataframe["RELATIVE_STRENGTH"] = (

            dataframe["STOCK_RETURN"]

            - dataframe["BENCHMARK_RETURN"]

        )

        dataframe["OUTPERFORM"] = (

            dataframe["RELATIVE_STRENGTH"] > 0

        )

        return dataframe

    ###########################################################################
    # Market Breadth
    ###########################################################################

    def calculate_market_breadth(
        self,
        dataframe: pd.DataFrame,
    ) -> dict[str, Any]:
        """
        Calculate market breadth statistics.
        """

        logger.info(
            "Calculating market breadth."
        )

        latest = (

            dataframe

            .sort_values("Date")

            .groupby("Symbol")

            .tail(1)

        )

        advancing = (

            latest["Close"]

            > latest["Close"].shift()

        ).sum()

        declining = (

            latest["Close"]

            < latest["Close"].shift()

        ).sum()

        unchanged = (

            latest["Close"]

            == latest["Close"].shift()

        ).sum()

        ratio = (

            advancing / declining

            if declining

            else float("inf")

        )

        self.market_breadth = {

            "ADVANCING": int(advancing),

            "DECLINING": int(declining),

            "UNCHANGED": int(unchanged),

            "ADVANCE_DECLINE_RATIO": ratio,

            "TOTAL": len(latest),

        }

        return self.market_breadth

    ###########################################################################
    # Data Quality
    ###########################################################################

    def calculate_data_quality(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calculate normalized data quality score.
        """

        logger.info(
            "Calculating data quality."
        )

        dataframe = dataframe.copy()

        missing = dataframe.isna().sum(axis=1)

        total_columns = len(dataframe.columns)

        dataframe["QUALITY_SCORE"] = (

            100

            * (

                1

                - (

                    missing

                    / total_columns

                )

            )

        ).clip(

            lower=0,

            upper=100,

        )

        return dataframe

    ###########################################################################
    # Storage
    ###########################################################################

    def save_market_data(
        self,
        dataframe: pd.DataFrame,
        *,
        pipeline_run_id: str,
    ) -> None:
        """
        Persist market data.
        """

        logger.info(
            "Saving market data."
        )

        self.storage.write_table(

            dataframe=dataframe,

            table="market_data",

            mode="replace",

            pipeline_run_id=pipeline_run_id,

            source="MarketDataEngine",

        )

        self.storage.save_parquet(

            dataframe,

            dataset="market_data",

        )

    ###########################################################################
    # Pipeline
    ###########################################################################

    def run_pipeline(
        self,
        *,
        pipeline_run_id: str,
        period: str = DEFAULT_PERIOD,
        interval: str = DEFAULT_INTERVAL,
    ) -> pd.DataFrame:
        """
        Execute complete market data pipeline.
        """

        logger.info(
            "Starting market pipeline."
        )

        dataframe = self.download_market_data(

            period=period,

            interval=interval,

        )

        dataframe = self.enrich_market_data(
            dataframe,
        )

        benchmark = self.download_benchmark(

            period=period,

            interval=interval,

        )

        dataframe = self.calculate_relative_strength(

            dataframe,

            benchmark,

        )

        dataframe = self.calculate_data_quality(
            dataframe,
        )

        self.calculate_market_breadth(
            dataframe,
        )

        self.save_market_data(

            dataframe,

            pipeline_run_id=pipeline_run_id,

        )

        self.market_data = dataframe

        self.last_refresh = datetime.utcnow()

        logger.info(
            "Market pipeline completed."
        )

        return dataframe

    ###########################################################################
    # Cache
    ###########################################################################

    def load_market_data(
        self,
    ) -> pd.DataFrame:
        """
        Load cached market data.
        """

        return self.storage.read_table(
            "market_data",
        )


    ###########################################################################
    # Summary
    ###########################################################################

    def summary(
        self,
    ) -> dict[str, Any]:
        """
        Return MarketDataEngine summary.
        """

        return {

            "engine": "MarketDataEngine",

            "symbols": len(
                self.universe,
            ),

            "universes": len(
                self.available_universes(),
            ),

            "large_cap": len(
                self.get_universe(
                    "LargeCap",
                )
            ),

            "mid_cap": len(
                self.get_universe(
                    "MidCap",
                )
            ),

            "small_cap": len(
                self.get_universe(
                    "SmallCap",
                )
            ),

            "benchmark": DEFAULT_BENCHMARK,

            "cached_rows": len(
                self.market_data,
            ),

            "last_refresh": self.last_refresh,

            "source": self.meta.get(
                "source",
                "Unknown",
            ),

        }