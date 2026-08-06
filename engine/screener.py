"""
engine/screener.py
==================

Institutional universe screening engine.

Responsibilities
----------------
* Filter non-tradable securities
* Apply liquidity rules
* Apply price rules
* Apply data quality rules
* Prepare eligible universe

This module does NOT generate trading signals.

Signal generation belongs to:
    engine.scanner
"""

from __future__ import annotations

###############################################################################
# Standard Library
###############################################################################
import logging
from dataclasses import dataclass
from typing import Any

###############################################################################
# Third Party
###############################################################################
import pandas as pd

###############################################################################
# Logger
###############################################################################

logger = logging.getLogger(__name__)


###############################################################################
# Screener Configuration
###############################################################################


@dataclass(
    frozen=True,
)
class ScreenerConfig:
    """
    Screener rules configuration.
    """

    ###########################################################################
    # Price Rules
    ###########################################################################

    minimum_price: float = 20.0

    maximum_price: float = 100000.0


    ###########################################################################
    # Liquidity Rules
    ###########################################################################

    minimum_volume: int = 100000

    minimum_turnover: float = 10000000.0


    ###########################################################################
    # Data Quality
    ###########################################################################

    minimum_history_days: int = 200

    minimum_quality_score: float = 70.0



###############################################################################
# Screener Engine
###############################################################################


class Screener:
    """
    Institutional stock universe screener.
    """

    ###########################################################################
    # Initialization
    ###########################################################################

    def __init__(
        self,
        config: ScreenerConfig | None = None,
    ) -> None:
        """
        Initialize screener.
        """

        self.config = (

            config

            if config is not None

            else ScreenerConfig()

        )

        self.input_rows: int = 0

        self.output_rows: int = 0

        self.filtered_rows: int = 0

        self.last_result = pd.DataFrame()


        logger.info(
            "Screener initialized."
        )


    ###########################################################################
    # Price Filter
    ###########################################################################

    def filter_price(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Filter securities based on price range.
        """

        logger.info(
            "Applying price filter."
        )

        config = self.config

        dataframe = dataframe.copy()

        dataframe = dataframe.loc[

            dataframe["Close"].between(
                config.minimum_price,
                config.maximum_price,
            )

        ]

        return dataframe


    ###########################################################################
    # Volume Filter
    ###########################################################################

    def filter_volume(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Filter securities based on minimum volume.
        """

        logger.info(
            "Applying volume filter."
        )

        dataframe = dataframe.copy()

        latest = (
            dataframe
            .sort_values("Date")
            .groupby("Symbol")
            .tail(1)
        )


        valid_symbols = latest.loc[
            latest["Volume"]
            >= self.config.minimum_volume,
            "Symbol",
        ]


        return dataframe.loc[
            dataframe["Symbol"]
            .isin(valid_symbols)
        ]


    ###########################################################################
    # Turnover Filter
    ###########################################################################

    def filter_turnover(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Filter based on traded value.

        Turnover = Close × Volume
        """

        logger.info(
            "Applying turnover filter."
        )

        dataframe = dataframe.copy()


        dataframe["TURNOVER"] = (

            dataframe["Close"]

            *

            dataframe["Volume"]

        )


        latest = (

            dataframe

            .sort_values("Date")

            .groupby("Symbol")

            .tail(1)

        )


        valid_symbols = latest.loc[

            latest["TURNOVER"]

            >= self.config.minimum_turnover,

            "Symbol",

        ]


        return dataframe.loc[

            dataframe["Symbol"]

            .isin(valid_symbols)

        ]


    ###########################################################################
    # Historical Data Filter
    ###########################################################################

    def filter_history(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Remove stocks with insufficient history.
        """

        logger.info(
            "Applying history filter."
        )


        history = (

            dataframe

            .groupby("Symbol")

            ["Date"]

            .count()

        )


        valid_symbols = history.loc[

            history

            >= self.config.minimum_history_days

        ].index


        return dataframe.loc[

            dataframe["Symbol"]

            .isin(valid_symbols)

        ]


    ###########################################################################
    # Quality Filter
    ###########################################################################

    def filter_quality(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Filter based on data quality score.
        """

        logger.info(
            "Applying quality filter."
        )


        if "QUALITY_SCORE" not in dataframe.columns:

            logger.warning(

                "QUALITY_SCORE missing. Skipping."

            )

            return dataframe


        return dataframe.loc[

            dataframe["QUALITY_SCORE"]

            >= self.config.minimum_quality_score

        ]


    ###########################################################################
    # Apply Filters
    ###########################################################################

    def apply_filters(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Execute complete screening pipeline.
        """

        logger.info(
            "Starting screening."
        )


        if dataframe.empty:

            raise ValueError(
                "Input dataframe is empty."
            )


        self.input_rows = len(dataframe)

        dataframe = self.filter_price(
            dataframe,
        )

        dataframe = self.filter_volume(
            dataframe,
        )

        dataframe = self.filter_turnover(
            dataframe,
        )

        dataframe = self.filter_history(
            dataframe,
        )

        dataframe = self.filter_quality(
            dataframe,
        )

        self.output_rows = len(dataframe)

        self.filtered_rows = (
            self.input_rows
            -
            self.output_rows
        )

        self.last_result = dataframe

        logger.info(

            "Screening completed. %d → %d rows.",
            self.input_rows,
            self.output_rows,
        )

        return dataframe


    ###########################################################################
    # Liquidity Score
    ###########################################################################

    def calculate_liquidity_score(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calculate liquidity score.

        Components:
        - Volume
        - Turnover
        - Trading consistency
        """

        logger.info(
            "Calculating liquidity score."
        )

        dataframe = dataframe.copy()

        latest = (

            dataframe
            .sort_values("Date")
            .groupby("Symbol")
            .tail(1)

        )


        latest["VOLUME_SCORE"] = (

            latest["Volume"]
            .rank(
                pct=True,
            )
            *
            100
        )


        latest["TURNOVER_SCORE"] = (

            latest["TURNOVER"]
            .rank(
                pct=True,
            )
            *
            100
        )


        latest["LIQUIDITY_SCORE"] = (

            latest["VOLUME_SCORE"]
            *
            0.5
            +
            latest["TURNOVER_SCORE"]
            *
            0.5

        )


        dataframe = dataframe.merge(

            latest[

                [
                    "Symbol",
                    "LIQUIDITY_SCORE",
                ]
            ],

            on="Symbol",
            how="left",
        )

        return dataframe


    ###########################################################################
    # Trend Readiness Score
    ###########################################################################

    def calculate_trend_score(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calculate trend readiness score.
        """

        logger.info(
            "Calculating trend score."
        )

        dataframe = dataframe.copy()


        if not {

            "EMA_50",
            "EMA_200",

        }.issubset(dataframe.columns):

            logger.warning(
                "Trend indicators missing."
            )

            dataframe["TREND_SCORE"] = 0

            return dataframe

        dataframe["TREND_SCORE"] = 0

        dataframe.loc[
            dataframe["Close"]
            >
            dataframe["EMA_50"],
            "TREND_SCORE",
        ] += 50


        dataframe.loc[
            dataframe["EMA_50"]
            >
            dataframe["EMA_200"],
            "TREND_SCORE",
        ] += 50


        return dataframe


    ###########################################################################
    # Composite Score
    ###########################################################################

    def calculate_screen_score(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calculate final screening score.
        """

        logger.info(
            "Calculating composite score."
        )

        dataframe = dataframe.copy()


        if "QUALITY_SCORE" not in dataframe.columns:

            dataframe["QUALITY_SCORE"] = 100


        dataframe["SCREEN_SCORE"] = (

            dataframe["LIQUIDITY_SCORE"]
            *
            0.40
            +
            dataframe["TREND_SCORE"]
            *
            0.40
            +
            dataframe["QUALITY_SCORE"]
            *
            0.20

        )

        return dataframe


    ###########################################################################
    # Ranking
    ###########################################################################

    def rank_universe(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Rank securities based on screening score.
        """

        logger.info(
            "Ranking universe."
        )

        latest = (
            dataframe
            .sort_values("Date")
            .groupby("Symbol")
            .tail(1)
        )


        latest["RANK"] = (

            latest["SCREEN_SCORE"]

            .rank(
                ascending=False,
                method="dense",
            )
            .astype(int)
        )


        dataframe = dataframe.merge(

            latest[

                [
                    "Symbol",
                    "RANK",
                ]
            ],

            on="Symbol",
            how="left",
        )

        return dataframe


    ###########################################################################
    # Top Universe
    ###########################################################################

    def select_top_universe(
        self,
        dataframe: pd.DataFrame,
        *,
        top_n: int = 500,
    ) -> pd.DataFrame:
        """
        Select highest ranked securities.
        """

        logger.info(
            "Selecting top %d universe.",
            top_n,
        )


        latest = (

            dataframe

            .sort_values("Date")

            .groupby("Symbol")

            .tail(1)

        )


        symbols = (

            latest

            .sort_values(
                "RANK",
            )

            .head(top_n)

            ["Symbol"]

            .tolist()

        )


        return dataframe.loc[

            dataframe["Symbol"]
            .isin(symbols)

        ]

    ###########################################################################
    # Complete Screening Pipeline
    ###########################################################################

    def run(
        self,
        dataframe: pd.DataFrame,
        *,
        top_n: int = 500,
    ) -> pd.DataFrame:
        """
        Execute complete screening pipeline.
        """

        logger.info(
            "Starting screener pipeline."
        )

        dataframe = self.apply_filters(
            dataframe,
        )

        dataframe = self.calculate_liquidity_score(
            dataframe,
        )

        dataframe = self.calculate_trend_score(
            dataframe,
        )

        dataframe = self.calculate_screen_score(
            dataframe,
        )

        dataframe = self.rank_universe(
            dataframe,
        )

        dataframe = self.select_top_universe(

            dataframe,
            top_n=top_n,

        )

        self.last_result = dataframe

        self.output_rows = len(dataframe)

        logger.info(
            "Screener pipeline completed. %d rows.",
            len(dataframe),
        )

        return dataframe


    ###########################################################################
    # Export Universe
    ###########################################################################

    def export_universe(
        self,
        dataframe: pd.DataFrame,
        path: str,
    ) -> None:
        """
        Export screened universe.
        """

        logger.info(
            "Exporting screened universe."
        )

        dataframe.to_csv(
            path,
            index=False,
        )


    ###########################################################################
    # Latest Universe
    ###########################################################################

    def latest_universe(
        self,
    ) -> pd.DataFrame:
        """
        Return latest screened universe.
        """

        return self.last_result.copy()


    ###########################################################################
    # Summary
    ###########################################################################

    def summary(
        self,
    ) -> dict[str, Any]:
        """
        Return screener summary.
        """

        return {

            "engine":
                "Screener",

            "input_rows":
                self.input_rows,

            "output_rows":
                self.output_rows,

            "filtered_rows":
                self.filtered_rows,

            "universe_size":
                self.last_result["Symbol"]

                .nunique()

                if not self.last_result.empty

                else 0,


            "config":

                {
                    "minimum_price":
                        self.config.minimum_price,

                    "minimum_volume":
                        self.config.minimum_volume,

                    "minimum_turnover":
                        self.config.minimum_turnover,

                    "minimum_history_days":
                        self.config.minimum_history_days,
                },
        }
