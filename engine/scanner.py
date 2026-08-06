"""
engine/scanner.py
=================

Institutional Swing Scanner.

Responsibilities
----------------
* Prepare market data
* Apply trading filters
* Calculate institutional scores
* Rank stocks
* Generate trade levels
* Return top-ranked swing opportunities

This module contains NO indicator calculations.

Indicators are provided by IndicatorEngine.
"""

from __future__ import annotations

###############################################################################
# Standard Library
###############################################################################
import logging
from typing import Any

###############################################################################
# Third Party
###############################################################################
import numpy as np
import pandas as pd

###############################################################################
# Local Imports
###############################################################################
from config import settings
from engine.data import MarketDataEngine
from engine.indicators import IndicatorEngine
from engine.storage import StorageManager

###############################################################################
# Logger
###############################################################################

logger = logging.getLogger(__name__)

###############################################################################
# Swing Scanner
###############################################################################


class SwingScanner:
    """
    Institutional Swing Scanner.

    Pipeline
    --------

        Market Data

            ↓

        Indicators

            ↓

        Filters

            ↓

        Scoring

            ↓

        Ranking

            ↓

        Trade Levels

            ↓

        Top Picks
    """

    ###########################################################################
    # Initialization
    ###########################################################################

    def __init__(
        self,
    ) -> None:
        """
        Initialize Swing Scanner.
        """

        self.settings = settings

        self.storage = StorageManager()

        self.data_engine = MarketDataEngine()

        self.indicators = IndicatorEngine()

        self.market_data = pd.DataFrame()

        self.scan_results = pd.DataFrame()

        logger.info(
            "SwingScanner initialized."
        )

    ###########################################################################
    # Data Preparation
    ###########################################################################

    def prepare_data(
        self,
        market_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Prepare market data for scanning.
        """

        if market_data.empty:

            raise ValueError(
                "Market data is empty."
            )

        dataframe = market_data.copy()

        dataframe = self.indicators.calculate_all(
            dataframe
        )

        self.market_data = dataframe

        logger.info(

            "Prepared %d rows.",

            len(dataframe),

        )

        return dataframe

    ###########################################################################
    # Filtering Engine
    ###########################################################################

    def apply_filters(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Apply all scanner filters.
        """

        logger.info(
            "Applying scanner filters."
        )

        dataframe = self._liquidity_filter(
            dataframe
        )

        dataframe = self._trend_filter(
            dataframe
        )

        dataframe = self._momentum_filter(
            dataframe
        )

        dataframe = self._volume_filter(
            dataframe
        )

        dataframe = self._risk_filter(
            dataframe
        )

        logger.info(
            "Remaining stocks: %d",
            dataframe["Symbol"].nunique(),
        )

        return dataframe


    ###########################################################################
    # Liquidity Filter
    ###########################################################################

    def _liquidity_filter(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Filter illiquid stocks.
        """

        minimum_volume = (
            self.settings.screener.min_volume
        )

        return dataframe.loc[

            dataframe["VOL_SMA_20"] >= minimum_volume

        ].copy()


    ###########################################################################
    # Trend Filter
    ###########################################################################

    def _trend_filter(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Bullish trend filter.
        """

        return dataframe.loc[

            (dataframe["Close"] > dataframe["EMA_20"])

            &

            (dataframe["EMA_20"] > dataframe["EMA_50"])

            &

            (dataframe["EMA_50"] > dataframe["EMA_200"])

            &

            (dataframe["ADX_14"] >= 20)

        ].copy()


    ###########################################################################
    # Momentum Filter
    ###########################################################################

    def _momentum_filter(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Momentum filter.
        """

        return dataframe.loc[

            (dataframe["RSI_14"] >= 55)

            &

            (dataframe["RSI_14"] <= 80)

            &

            (dataframe["MACD"] > dataframe["MACD_SIGNAL"])

            &

            (dataframe["ROC_20"] > 0)

        ].copy()


    ###########################################################################
    # Volume Filter
    ###########################################################################

    def _volume_filter(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Volume confirmation.
        """

        return dataframe.loc[

            dataframe["RVOL_20"] >= 1.20

        ].copy()


    ###########################################################################
    # Risk Filter
    ###########################################################################

    def _risk_filter(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Remove highly volatile stocks.
        """

        return dataframe.loc[

            dataframe["ATR_PCT"] <= 8.0

        ].copy()


    ###########################################################################
    # Scoring Engine
    ###########################################################################

    def calculate_scores(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calculate institutional score.
        """

        logger.info(
            "Calculating institutional scores."
        )

        dataframe = dataframe.copy()

        dataframe["TREND_SCORE"] = self._trend_score(
            dataframe
        )

        dataframe["MOMENTUM_SCORE"] = self._momentum_score(
            dataframe
        )

        dataframe["VOLUME_SCORE"] = self._volume_score(
            dataframe
        )

        dataframe["VOLATILITY_SCORE"] = self._volatility_score(
            dataframe
        )

        dataframe["RISK_SCORE"] = self._risk_score(
            dataframe
        )

        dataframe["TOTAL_SCORE"] = (

            dataframe["TREND_SCORE"]

            + dataframe["MOMENTUM_SCORE"]

            + dataframe["VOLUME_SCORE"]

            + dataframe["VOLATILITY_SCORE"]

            + dataframe["RISK_SCORE"]

        )

        return dataframe


    ###########################################################################
    # Trend Score (30)
    ###########################################################################

    def _trend_score(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.Series:
        """
        Trend score.
        """

        score = pd.Series(
            0.0,
            index=dataframe.index,
        )

        score += np.where(
            dataframe["Close"] > dataframe["EMA_20"],
            8,
            0,
        )

        score += np.where(
            dataframe["EMA_20"] > dataframe["EMA_50"],
            8,
            0,
        )

        score += np.where(
            dataframe["EMA_50"] > dataframe["EMA_200"],
            8,
            0,
        )

        score += np.where(
            dataframe["ADX_14"] >= 25,
            6,
            0,
        )

        return score


    ###########################################################################
    # Momentum Score (20)
    ###########################################################################

    def _momentum_score(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.Series:
        """
        Momentum score.
        """

        score = pd.Series(
            0.0,
            index=dataframe.index,
        )

        score += np.where(
            dataframe["RSI_14"].between(55, 70),
            8,
            0,
        )

        score += np.where(
            dataframe["MACD"] > dataframe["MACD_SIGNAL"],
            6,
            0,
        )

        score += np.where(
            dataframe["ROC_20"] > 0,
            6,
            0,
        )

        return score


    ###########################################################################
    # Volume Score (10)
    ###########################################################################

    def _volume_score(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.Series:
        """
        Volume score.
        """

        score = pd.Series(
            0.0,
            index=dataframe.index,
        )

        score += np.where(
            dataframe["RVOL_20"] >= 1.20,
            5,
            0,
        )

        score += np.where(
            dataframe["OBV"].diff() > 0,
            5,
            0,
        )

        return score


    ###########################################################################
    # Volatility Score (10)
    ###########################################################################

    def _volatility_score(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.Series:
        """
        Volatility score.
        """

        score = pd.Series(
            0.0,
            index=dataframe.index,
        )

        score += np.where(
            dataframe["ATR_PCT"] <= 5,
            5,
            0,
        )

        score += np.where(
            dataframe["BB_WIDTH"] < 0.25,
            5,
            0,
        )

        return score


    ###########################################################################
    # Risk Score (5)
    ###########################################################################

    def _risk_score(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.Series:
        """
        Risk score.
        """

        score = pd.Series(
            5.0,
            index=dataframe.index,
        )

        score -= np.where(
            dataframe["ATR_PCT"] > 8,
            3,
            0,
        )

        score -= np.where(
            dataframe["GAP_PCT"].abs() > 5,
            2,
            0,
        )

        return score.clip(
            lower=0,
        )


    ###########################################################################
    # Ranking Engine
    ###########################################################################

    def rank_stocks(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Rank stocks by institutional score.
        """

        logger.info(
            "Ranking stocks."
        )

        dataframe = dataframe.copy()

        dataframe["RANK"] = (

            dataframe["TOTAL_SCORE"]

            .rank(

                ascending=False,

                method="dense",

            )

            .astype(int)

        )

        dataframe = dataframe.sort_values(

            [

                "RANK",

                "TOTAL_SCORE",

            ],

            ascending=[

                True,

                False,

            ],

        )

        return dataframe


    ###########################################################################
    # Signal Engine
    ###########################################################################

    def generate_signals(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate trading signals.
        """

        dataframe = dataframe.copy()

        dataframe["SIGNAL"] = np.select(

            [

                dataframe["TOTAL_SCORE"] >= 55,

                dataframe["TOTAL_SCORE"] >= 45,

                dataframe["TOTAL_SCORE"] >= 35,

            ],

            [

                "BUY",

                "WATCH",

                "HOLD",

            ],

            default="AVOID",

        )

        return dataframe


    ###########################################################################
    # Trade Levels
    ###########################################################################

    def generate_trade_levels(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate entry, stop-loss and target.
        """

        logger.info(
            "Generating trade levels."
        )

        dataframe = dataframe.copy()

        dataframe["ENTRY"] = dataframe["Close"]

        dataframe["STOP_LOSS"] = (

            dataframe["ENTRY"]

            - (1.5 * dataframe["ATR_14"])

        )

        dataframe["TARGET"] = (

            dataframe["ENTRY"]

            + (3.0 * dataframe["ATR_14"])

        )

        dataframe["RISK"] = (

            dataframe["ENTRY"]

            - dataframe["STOP_LOSS"]

        )

        dataframe["REWARD"] = (

            dataframe["TARGET"]

            - dataframe["ENTRY"]

        )

        dataframe["RISK_REWARD"] = (

            dataframe["REWARD"]

            / dataframe["RISK"]

        ).round(2)

        return dataframe


    ###########################################################################
    # Selection
    ###########################################################################

    def select_top(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Return top ranked stocks.
        """

        logger.info(
            "Selecting top %d stocks.",
            self.settings.scanner.top_stocks,
        )

        dataframe = dataframe.copy()

        dataframe = dataframe.nsmallest(

            self.settings.scanner.top_stocks,

            "RANK",

        )

        self.scan_results = dataframe

        return dataframe


    ###########################################################################
    # Scanner Pipeline
    ###########################################################################

    def scan_market(
        self,
        market_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Execute complete market scanning pipeline.

        Pipeline
        --------
            Prepare Data
                ↓
            Apply Filters
                ↓
            Calculate Scores
                ↓
            Generate Signals
                ↓
            Rank Stocks
                ↓
            Generate Trade Levels
                ↓
            Select Top Stocks
        """

        logger.info(
            "Starting market scan."
        )

        dataframe = self.prepare_data(
            market_data
        )

        dataframe = self.apply_filters(
            dataframe
        )

        dataframe = self.calculate_scores(
            dataframe
        )

        dataframe = self.generate_signals(
            dataframe
        )

        dataframe = self.rank_stocks(
            dataframe
        )

        dataframe = self.generate_trade_levels(
            dataframe
        )

        dataframe = self.select_top(
            dataframe
        )

        self.scan_results = dataframe

        logger.info(
            "Market scan completed successfully."
        )

        return dataframe


    ###########################################################################
    # Persistence
    ###########################################################################

    def save_results(
        self,
        dataframe: pd.DataFrame,
    ) -> None:
        """
        Save scan results.
        """

        logger.info(
            "Saving scan results."
        )

        self.storage.write_table(

            dataframe=dataframe,

            table="scan_results",

            mode="replace",

            source="scanner",

        )

        self.storage.save_parquet(

            dataframe,

            dataset="scan_results",

        )

        self.storage.save_excel(

            dataframe,

            filename="scan_results.xlsx",

        )


    def load_results(
        self,
    ) -> pd.DataFrame:
        """
        Load previous scan results.
        """

        return self.storage.read_table(
            "scan_results"
        )


    ###########################################################################
    # Utilities
    ###########################################################################

    def get_buy_signals(
        self,
    ) -> pd.DataFrame:
        """
        Return BUY signals only.
        """

        if self.scan_results.empty:

            return pd.DataFrame()

        return self.scan_results.loc[

            self.scan_results["SIGNAL"] == "BUY"

        ].copy()


    def get_watchlist(
        self,
    ) -> pd.DataFrame:
        """
        Return WATCH signals.
        """

        if self.scan_results.empty:

            return pd.DataFrame()

        return self.scan_results.loc[

            self.scan_results["SIGNAL"] == "WATCH"

        ].copy()


    ###########################################################################
    # Summary
    ###########################################################################

    def summary(
        self,
    ) -> dict[str, Any]:
        """
        Scanner summary.
        """

        buy_count = 0
        watch_count = 0

        if not self.scan_results.empty:

            buy_count = int(

                (
                    self.scan_results["SIGNAL"]
                    == "BUY"
                ).sum()

            )

            watch_count = int(

                (
                    self.scan_results["SIGNAL"]
                    == "WATCH"
                ).sum()

            )

        return {

            "engine": "SwingScanner",

            "market_rows": len(
                self.market_data
            ),

            "scan_rows": len(
                self.scan_results
            ),

            "buy_signals": buy_count,

            "watch_signals": watch_count,

            "top_stocks": self.settings.scanner.top_stocks,

        }
