"""
engine/backtest.py
==================

Institutional Backtesting Engine.

Responsibilities
----------------
* Prepare scanner results
* Generate simulated trades
* Calculate portfolio performance
* Produce institutional backtest statistics

This module does NOT:

    * Download market data
    * Calculate indicators
    * Scan the market

Those responsibilities belong to:

    data.py
    indicators.py
    scanner.py
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
from engine.storage import StorageManager

###############################################################################
# Logger
###############################################################################

logger = logging.getLogger(__name__)

###############################################################################
# Backtest Engine
###############################################################################


class BacktestEngine:
    """
    Institutional Backtesting Engine.

    Pipeline
    --------

        Scanner Results

            ↓

        Trade Generation

            ↓

        Trade Simulation

            ↓

        Portfolio Statistics

            ↓

        Equity Curve

            ↓

        Reports
    """

    ###########################################################################
    # Initialization
    ###########################################################################

    def __init__(
        self,
    ) -> None:
        """
        Initialize Backtest Engine.
        """

        self.settings = settings

        self.storage = StorageManager()

        self.market_data = pd.DataFrame()

        self.trade_log = pd.DataFrame()

        self.equity_curve = pd.DataFrame()

        self.statistics: dict[str, Any] = {}

        logger.info(
            "BacktestEngine initialized."
        )

    ###########################################################################
    # Data Preparation
    ###########################################################################

    def prepare_data(
        self,
        market_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Prepare historical market data for backtesting.
        """

        if market_data.empty:

            raise ValueError(
                "Market data is empty."
            )

        dataframe = market_data.copy()

        dataframe = dataframe.sort_values(

            [

                "Symbol",

                "Date",

            ]

        )

        dataframe = dataframe.reset_index(

            drop=True,

        )

        self.market_data = dataframe

        logger.info(

            "Prepared %d rows.",

            len(dataframe),

        )

        return dataframe

    ###########################################################################
    # Trade Generation
    ###########################################################################

    def generate_trades(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate simulated trades from scanner signals.
        """

        logger.info(
            "Generating trades."
        )

        if dataframe.empty:

            return pd.DataFrame()

        dataframe = dataframe.copy()

        dataframe = dataframe.loc[

            dataframe["SIGNAL"] == "BUY"

        ].copy()

        dataframe["ENTRY_DATE"] = dataframe["Date"]

        dataframe["ENTRY_PRICE"] = dataframe["ENTRY"]

        dataframe["STOP_PRICE"] = dataframe["STOP_LOSS"]

        dataframe["TARGET_PRICE"] = dataframe["TARGET"]

        dataframe["STATUS"] = "OPEN"

        dataframe["EXIT_DATE"] = pd.NaT

        dataframe["EXIT_PRICE"] = np.nan

        dataframe["EXIT_REASON"] = ""

        dataframe["HOLDING_DAYS"] = 0

        self.trade_log = dataframe

        logger.info(

            "Generated %d trades.",

            len(dataframe),

        )

        return dataframe


    ###########################################################################
    # Trade Simulation
    ###########################################################################

    def simulate_trades(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Simulate trade execution.

        Exit priority

            1. Stop Loss

            2. Target

            3. Time Exit
        """

        logger.info(
            "Simulating trades."
        )

        if dataframe.empty:

            return dataframe

        dataframe = dataframe.copy()

        max_holding = self.settings.backtest.max_holding_days

        for index, row in dataframe.iterrows():

            entry = row["ENTRY_PRICE"]

            stop = row["STOP_PRICE"]

            target = row["TARGET_PRICE"]

            close = row["Close"]

            holding = min(
                row.get(
                    "HOLDING_DAYS",
                    0,
                ),
                max_holding,
            )

            exit_price = close

            exit_reason = "TIME"

            if row["Low"] <= stop:

                exit_price = stop

                exit_reason = "STOP"

            elif row["High"] >= target:

                exit_price = target

                exit_reason = "TARGET"

            dataframe.at[
                index,
                "EXIT_PRICE",
            ] = exit_price

            dataframe.at[
                index,
                "EXIT_REASON",
            ] = exit_reason

            dataframe.at[
                index,
                "HOLDING_DAYS",
            ] = holding

            dataframe.at[
                index,
                "STATUS",
            ] = "CLOSED"

        self.trade_log = dataframe

        return dataframe


    ###########################################################################
    # Position Sizing
    ###########################################################################

    def calculate_position_size(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calculate quantity based on risk.
        """

        dataframe = dataframe.copy()

        capital = self.settings.backtest.initial_capital

        risk_pct = self.settings.backtest.risk_per_trade

        risk_amount = capital * risk_pct

        dataframe["RISK_PER_SHARE"] = (

            dataframe["ENTRY_PRICE"]

            - dataframe["STOP_PRICE"]

        ).abs()

        dataframe["QUANTITY"] = (

            risk_amount

            / dataframe["RISK_PER_SHARE"]

        )

        dataframe["QUANTITY"] = (

            dataframe["QUANTITY"]

            .fillna(0)

            .astype(int)

        )

        return dataframe


    ###########################################################################
    # Trade Validation
    ###########################################################################

    def validate_trades(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Validate generated trades.
        """

        dataframe = dataframe.copy()

        dataframe = dataframe.dropna(

            subset=[

                "ENTRY_PRICE",

                "STOP_PRICE",

                "TARGET_PRICE",

            ]

        )

        dataframe = dataframe.loc[

            dataframe["ENTRY_PRICE"] > 0

        ]

        dataframe = dataframe.loc[

            dataframe["STOP_PRICE"]

            < dataframe["ENTRY_PRICE"]

        ]

        dataframe = dataframe.loc[

            dataframe["TARGET_PRICE"]

            > dataframe["ENTRY_PRICE"]

        ]

        return dataframe


    ###########################################################################
    # Performance Engine
    ###########################################################################

    def calculate_trade_statistics(
        self,
        dataframe: pd.DataFrame,
    ) -> dict[str, Any]:
        """
        Calculate trade statistics.
        """

        logger.info(
            "Calculating trade statistics."
        )

        if dataframe.empty:

            return {}

        dataframe = dataframe.copy()

        dataframe["GROSS_PNL"] = (

            dataframe["EXIT_PRICE"]

            - dataframe["ENTRY_PRICE"]

        ) * dataframe["QUANTITY"]

        commission_rate = (
            self.settings.backtest.commission
        )

        slippage_rate = (
            self.settings.backtest.slippage
        )

        dataframe["COMMISSION"] = (

            dataframe["ENTRY_PRICE"]

            * dataframe["QUANTITY"]

            * commission_rate

        )

        dataframe["SLIPPAGE"] = (

            dataframe["ENTRY_PRICE"]

            * dataframe["QUANTITY"]

            * slippage_rate

        )

        dataframe["NET_PNL"] = (

            dataframe["GROSS_PNL"]

            - dataframe["COMMISSION"]

            - dataframe["SLIPPAGE"]

        )

        dataframe["RETURN_PCT"] = (

            (

                dataframe["EXIT_PRICE"]

                - dataframe["ENTRY_PRICE"]

            )

            / dataframe["ENTRY_PRICE"]

            * 100

        )

        dataframe["WIN"] = (

            dataframe["NET_PNL"] > 0

        )

        self.trade_log = dataframe

        wins = dataframe.loc[
            dataframe["WIN"]
        ]

        losses = dataframe.loc[
            ~dataframe["WIN"]
        ]

        statistics = {

            "TOTAL_TRADES": len(dataframe),

            "WINNERS": len(wins),

            "LOSERS": len(losses),

            "WIN_RATE": (

                len(wins)

                / len(dataframe)

                * 100

                if len(dataframe)

                else 0

            ),

            "TOTAL_GROSS_PNL": dataframe[
                "GROSS_PNL"
            ].sum(),

            "TOTAL_NET_PNL": dataframe[
                "NET_PNL"
            ].sum(),

            "AVG_WIN": (

                wins["NET_PNL"].mean()

                if not wins.empty

                else 0

            ),

            "AVG_LOSS": (

                losses["NET_PNL"].mean()

                if not losses.empty

                else 0

            ),

            "BEST_TRADE": dataframe[
                "NET_PNL"
            ].max(),

            "WORST_TRADE": dataframe[
                "NET_PNL"
            ].min(),

            "AVG_RETURN_PCT": dataframe[
                "RETURN_PCT"
            ].mean(),

        }

        self.statistics.update(
            statistics
        )

        return statistics


    ###########################################################################
    # Portfolio Statistics
    ###########################################################################

    def calculate_portfolio_statistics(
        self,
        dataframe: pd.DataFrame,
    ) -> dict[str, Any]:
        """
        Calculate portfolio statistics.
        """

        logger.info(
            "Calculating portfolio statistics."
        )

        if dataframe.empty:

            return {}

        equity = (

            self.settings.backtest.initial_capital

            + dataframe["NET_PNL"].cumsum()

        )

        running_max = equity.cummax()

        drawdown = equity - running_max

        drawdown_pct = (

            drawdown

            / running_max

            * 100

        )

        total_return = (

            (

                equity.iloc[-1]

                - self.settings.backtest.initial_capital

            )

            / self.settings.backtest.initial_capital

            * 100

        )

        portfolio = {

            "INITIAL_CAPITAL":
            self.settings.backtest.initial_capital,

            "FINAL_CAPITAL":
            equity.iloc[-1],

            "TOTAL_RETURN_PCT":
            total_return,

            "MAX_DRAWDOWN":
            drawdown.min(),

            "MAX_DRAWDOWN_PCT":
            drawdown_pct.min(),

            "PROFIT_FACTOR": (

                abs(

                    dataframe.loc[
                        dataframe["NET_PNL"] > 0,
                        "NET_PNL",
                    ].sum()

                )

                /

                abs(

                    dataframe.loc[
                        dataframe["NET_PNL"] < 0,
                        "NET_PNL",
                    ].sum()

                )

                if (

                    dataframe["NET_PNL"] < 0

                ).any()

                else np.inf

            ),

        }

        self.statistics.update(
            portfolio
        )

        return portfolio


    ###########################################################################
    # Equity Curve
    ###########################################################################

    def build_equity_curve(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Build equity curve.
        """

        logger.info(
            "Building equity curve."
        )

        equity = pd.DataFrame()

        equity["DATE"] = dataframe[
            "EXIT_DATE"
        ]

        equity["NET_PNL"] = dataframe[
            "NET_PNL"
        ]

        equity["EQUITY"] = (

            self.settings.backtest.initial_capital

            + equity["NET_PNL"].cumsum()

        )

        equity["RUNNING_MAX"] = (

            equity["EQUITY"]

            .cummax()

        )

        equity["DRAWDOWN"] = (

            equity["EQUITY"]

            - equity["RUNNING_MAX"]

        )

        equity["DRAWDOWN_PCT"] = (

            equity["DRAWDOWN"]

            / equity["RUNNING_MAX"]

            * 100

        )

        self.equity_curve = equity

        return equity


    ###########################################################################
    # Backtest Pipeline
    ###########################################################################

    def run_backtest(
        self,
        market_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Execute complete backtest pipeline.

        Pipeline
        --------
            Prepare Data
                ↓
            Generate Trades
                ↓
            Validate Trades
                ↓
            Position Sizing
                ↓
            Simulate Trades
                ↓
            Trade Statistics
                ↓
            Portfolio Statistics
                ↓
            Equity Curve
        """

        logger.info(
            "Starting backtest."
        )

        dataframe = self.prepare_data(
            market_data,
        )

        dataframe = self.generate_trades(
            dataframe,
        )

        dataframe = self.validate_trades(
            dataframe,
        )

        dataframe = self.calculate_position_size(
            dataframe,
        )

        dataframe = self.simulate_trades(
            dataframe,
        )

        self.calculate_trade_statistics(
            dataframe,
        )

        self.calculate_portfolio_statistics(
            dataframe,
        )

        self.build_equity_curve(
            dataframe,
        )

        self.trade_log = dataframe

        logger.info(
            "Backtest completed successfully."
        )

        return dataframe


    ###########################################################################
    # Persistence
    ###########################################################################

    def save_results(
        self,
    ) -> None:
        """
        Save backtest outputs.
        """

        logger.info(
            "Saving backtest results."
        )

        if not self.trade_log.empty:

            self.storage.write_table(

                dataframe=self.trade_log,

                table="backtest_trades",

                mode="replace",

                source="backtest",

            )

            self.storage.save_parquet(

                self.trade_log,

                dataset="backtest_trades",

            )

            self.storage.save_excel(

                self.trade_log,

                filename="backtest_trades.xlsx",

            )

        if not self.equity_curve.empty:

            self.storage.write_table(

                dataframe=self.equity_curve,

                table="equity_curve",

                mode="replace",

                source="backtest",

            )

            self.storage.save_parquet(

                self.equity_curve,

                dataset="equity_curve",

            )

            self.storage.save_excel(

                self.equity_curve,

                filename="equity_curve.xlsx",

            )


    ###########################################################################
    # Load Results
    ###########################################################################

    def load_trade_log(
        self,
    ) -> pd.DataFrame:
        """
        Load stored trade log.
        """

        return self.storage.read_table(
            "backtest_trades",
        )


    def load_equity_curve(
        self,
    ) -> pd.DataFrame:
        """
        Load stored equity curve.
        """

        return self.storage.read_table(
            "equity_curve",
        )


    ###########################################################################
    # Export
    ###########################################################################

    def export_statistics(
        self,
    ) -> pd.DataFrame:
        """
        Export statistics as DataFrame.
        """

        return pd.DataFrame(
            [self.statistics]
        )


    ###########################################################################
    # Summary
    ###########################################################################

    def summary(
        self,
    ) -> dict[str, Any]:
        """
        Return engine summary.
        """

        return {

            "engine": "BacktestEngine",

            "market_rows": len(
                self.market_data,
            ),

            "trade_count": len(
                self.trade_log,
            ),

            "equity_points": len(
                self.equity_curve,
            ),

            "initial_capital": self.settings.backtest.initial_capital,

            "final_capital": self.statistics.get(
                "FINAL_CAPITAL",
                0,
            ),

            "total_return_pct": self.statistics.get(
                "TOTAL_RETURN_PCT",
                0,
            ),

            "win_rate": self.statistics.get(
                "WIN_RATE",
                0,
            ),

            "profit_factor": self.statistics.get(
                "PROFIT_FACTOR",
                0,
            ),

            "max_drawdown_pct": self.statistics.get(
                "MAX_DRAWDOWN_PCT",
                0,
            ),

        }
