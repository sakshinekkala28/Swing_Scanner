"""
workflows/run_backtest.py
=========================

Standalone backtest workflow.

Responsibilities
----------------
* Load historical market data
* Execute BacktestEngine
* Generate performance statistics
* Export backtest outputs

Business logic remains inside:

    engine/backtest.py
"""

from __future__ import annotations

###############################################################################
# Standard Library
###############################################################################
import logging
from datetime import datetime
from typing import Any

###############################################################################
# Third Party
###############################################################################
import pandas as pd

###############################################################################
# Local Imports
###############################################################################
from config import settings
from engine.backtest import BacktestEngine
from engine.reports import ReportEngine

###############################################################################
# Logger
###############################################################################

logger = logging.getLogger(__name__)


###############################################################################
# Backtest Workflow
###############################################################################


class BacktestWorkflow:
    """
    Standalone backtest execution workflow.
    """

    ###########################################################################
    # Initialization
    ###########################################################################

    def __init__(
        self,
        *,
        backtest_engine: BacktestEngine | None = None,
        report_engine: ReportEngine | None = None,
    ) -> None:
        """
        Initialize backtest workflow.
        """

        self.settings = settings

        self.backtest_engine = (

            backtest_engine

            if backtest_engine is not None

            else BacktestEngine()

        )

        self.report_engine = (

            report_engine

            if report_engine is not None

            else ReportEngine()

        )


        self.market_data = pd.DataFrame()

        self.trade_log = pd.DataFrame()

        self.equity_curve = pd.DataFrame()

        self.statistics: dict[str, Any] = {}


        self.started: datetime | None = None

        self.finished: datetime | None = None


        logger.info(
            "BacktestWorkflow initialized."
        )


    ###########################################################################
    # Data Loading
    ###########################################################################

    def load_data(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Load backtest input data.
        """

        if dataframe.empty:

            raise ValueError(
                "Backtest input data is empty."
            )

        self.market_data = dataframe.copy()

        return self.market_data


    ###########################################################################
    # Backtest Execution
    ###########################################################################

    def run_backtest(
        self,
        dataframe: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Execute backtest.
        """

        if dataframe is None:

            dataframe = self.market_data


        if dataframe.empty:

            raise ValueError(
                "No data available."
            )


        self.trade_log = (

            self.backtest_engine.run_backtest(

                dataframe,

            )

        )


        self.equity_curve = (

            self.backtest_engine.equity_curve

        )


        self.statistics = (

            self.backtest_engine.statistics.copy()

        )

        return self.trade_log

    ###########################################################################
    # Report Generation
    ###########################################################################

    def generate_reports(
        self,
    ) -> dict[str, pd.DataFrame]:
        """
        Generate backtest reports.
        """

        if self.trade_log.empty:

            raise ValueError(
                "No backtest results available."
            )


        self.report_outputs = (

            self.report_engine.generate_backtest_reports(

                trade_log=self.trade_log,

                equity_curve=self.equity_curve,

                statistics=self.statistics,

            )

        )


        return self.report_outputs


    ###########################################################################
    # Results
    ###########################################################################

    def results(
        self,
    ) -> dict[str, Any]:
        """
        Return workflow results.
        """

        return {

            "market_data":
                self.market_data,

            "trade_log":
                self.trade_log,

            "equity_curve":
                self.equity_curve,

            "statistics":
                self.statistics,

            "reports":
                getattr(
                    self,
                    "report_outputs",
                    {},
                ),
        }

    ###########################################################################
    # Master Run
    ###########################################################################

    def run(
        self,
        dataframe: pd.DataFrame,
    ) -> dict[str, Any]:
        """
        Execute complete backtest workflow.
        """

        self.started = datetime.utcnow()

        try:

            self.load_data(
                dataframe,
            )

            self.run_backtest()

            self.generate_reports()

            self.finished = datetime.utcnow()


            return self.results()


        except Exception as exc:

            logger.exception(
                "Backtest workflow failed: %s",
                exc,
            )

            raise


    ###########################################################################
    # Execution Time
    ###########################################################################

    def execution_time(
        self,
    ) -> float:
        """
        Return execution time.
        """

        if not self.started or not self.finished:

            return 0.0


        return (

            self.finished

            -

            self.started

        ).total_seconds()


    ###########################################################################
    # Summary
    ###########################################################################

    def summary(
        self,
    ) -> dict[str, Any]:
        """
        Return workflow summary.
        """

        return {

            "workflow":
                "BacktestWorkflow",

            "trades":
                len(self.trade_log),

            "equity_points":
                len(self.equity_curve),

            "statistics":
                len(self.statistics),

            "reports":
                len(
                    getattr(
                        self,
                        "report_outputs",
                        {},
                    )
                ),

            "execution_seconds":
                self.execution_time(),

        }
