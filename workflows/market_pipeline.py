"""
workflows/market_pipeline.py
============================

Master orchestration pipeline for the Swing Scanner platform.

Responsibilities
----------------
* Execute MarketDataEngine
* Execute IndicatorEngine
* Execute SwingScanner
* Execute BacktestEngine
* Execute ReportEngine
* Return complete pipeline outputs

This module contains NO business logic.

All calculations remain inside their respective engines.
"""

from __future__ import annotations

###############################################################################
# Standard Library
###############################################################################
import logging
from datetime import datetime, UTC
from time import perf_counter
from typing import Any

###############################################################################
# Third Party
###############################################################################
import pandas as pd

###############################################################################
# Local Imports
###############################################################################
from config import settings
from core.pipeline_result import PipelineResult
from engine.backtest import BacktestEngine
from engine.data import MarketDataEngine
from engine.indicators import IndicatorEngine
from engine.reports import ReportEngine
from engine.scanner import SwingScanner


###############################################################################
# Logger
###############################################################################

logger = logging.getLogger(__name__)

###############################################################################
# Market Pipeline
###############################################################################


class MarketPipeline:
    """
    Master Swing Scanner pipeline.

    Pipeline
    --------

        MarketDataEngine

            ↓

        IndicatorEngine

            ↓

        SwingScanner

            ↓

        BacktestEngine

            ↓

        ReportEngine
    """

    ###########################################################################
    # Initialization
    ###########################################################################

    def __init__(
        self,
        *,
        market_engine: MarketDataEngine | None = None,
        indicator_engine: IndicatorEngine | None = None,
        scanner: SwingScanner | None = None,
        backtest: BacktestEngine | None = None,
        reports: ReportEngine | None = None,
    ) -> None:
        """
        Initialize pipeline.

        Dependencies can be injected for:
        - testing
        - alternate data providers
        - custom engines
        """

        self.settings = settings

        #######################################################################
        # Engines
        #######################################################################

        self.market_engine = (

            market_engine

            if market_engine is not None

            else MarketDataEngine()

        )

        self.indicator_engine = (

            indicator_engine

            if indicator_engine is not None

            else IndicatorEngine()

        )

        self.scanner = (

            scanner

            if scanner is not None

            else SwingScanner()

        )

        self.backtest = (

            backtest

            if backtest is not None

            else BacktestEngine()

        )

        self.reports = (

            reports

            if reports is not None

            else ReportEngine()

        )

        #######################################################################
        # Pipeline Data State
        #######################################################################

        self.market_data = pd.DataFrame()

        self.indicator_data = pd.DataFrame()

        self.scan_results = pd.DataFrame()

        self.trade_log = pd.DataFrame()

        self.equity_curve = pd.DataFrame()

        #######################################################################
        # Results
        #######################################################################

        self.statistics: dict[str, Any] = {}

        self.report_outputs: dict[str, pd.DataFrame] = {}

        #######################################################################
        # Execution Metadata
        #######################################################################

        self.pipeline_run_id: str | None = None

        self.pipeline_started: datetime | None = None

        self.pipeline_finished: datetime | None = None

        self.failed_stage: str | None = None

        self.progress_callback = None

        self.metrics: dict[str, float] = {}

        logger.info(
            "MarketPipeline initialized."
        )

    def update_progress(
        self,
        message: str,
        progress: float,
    ) -> None:
        """
        Update pipeline progress.
        """

        logger.info(
            message
        )

        if self.progress_callback:

            self.progress_callback(
                message,
                progress,
            )

    ###########################################################################
    # Engine Access
    ###########################################################################

    @property
    def engines(
        self,
    ) -> dict[str, Any]:
        """
        Return all engine instances.
        """

        return {

            "market": self.market_engine,

            "indicator": self.indicator_engine,

            "scanner": self.scanner,

            "backtest": self.backtest,

            "reports": self.reports,

        }

    ###########################################################################
    # Pipeline State
    ###########################################################################

    @property
    def is_complete(
        self,
    ) -> bool:
        """
        Return pipeline completion status.
        """

        return (

            self.pipeline_finished

            is not None

        )

    ###########################################################################
    # Reset
    ###########################################################################

    def reset(
        self,
    ) -> None:
        """
        Reset pipeline state.
        """

        logger.info(
            "Resetting pipeline."
        )

        self.market_data = pd.DataFrame()

        self.indicator_data = pd.DataFrame()

        self.scan_results = pd.DataFrame()

        self.trade_log = pd.DataFrame()

        self.equity_curve = pd.DataFrame()

        self.statistics.clear()

        self.report_outputs.clear()

        self.pipeline_started = None

        self.pipeline_finished = None

        self.pipeline_run_id = None

        self.failed_stage = None

        self.metrics.clear()


    ###########################################################################
    # Market Data Stage
    ###########################################################################

    def run_market_data(
        self,
        *,
        pipeline_run_id: str,
        period: str = "10y",
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Execute market data stage.
        """

        logger.info(
            "Running market data stage."
        )

        self.market_data = (

            self.market_engine.run_pipeline(

                pipeline_run_id=pipeline_run_id,

                period=period,

                interval=interval,

            )

        )

        logger.info(

            "Market data completed: %d rows.",

            len(self.market_data),

        )

        return self.market_data


    ###########################################################################
    # Indicator Stage
    ###########################################################################

    def run_indicators(
        self,
        dataframe: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Execute technical indicator calculation.
        """

        logger.info(
            "Running indicator stage."
        )

        if dataframe is None:

            dataframe = self.market_data


        if dataframe.empty:

            raise ValueError(
                "Market data is empty. "
                "Run market data stage first."
            )


        self.indicator_data = (

            self.indicator_engine.calculate(

                dataframe,

            )

        )


        logger.info(

            "Indicators completed: %d rows.",

            len(self.indicator_data),

        )


        return self.indicator_data


    ###########################################################################
    # Scanner Stage
    ###########################################################################

    def run_scanner(
        self,
        dataframe: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Execute swing scanner.
        """

        logger.info(
            "Running scanner stage."
        )


        if dataframe is None:

            dataframe = self.indicator_data


        if dataframe.empty:

            raise ValueError(
                "Indicator data is empty. "
                "Run indicator stage first."
            )


        self.scan_results = (

            self.scanner.scan_market(

                dataframe,

            )

        )


        logger.info(

            "Scanner completed: %d results.",

            len(self.scan_results),

        )


        return self.scan_results


    ###########################################################################
    # Stage Status
    ###########################################################################

    def stage_status(
        self,
    ) -> dict[str, bool]:
        """
        Return pipeline stage completion status.
        """

        return {

            "market_data":
                not self.market_data.empty,

            "indicators":
                not self.indicator_data.empty,

            "scanner":
                not self.scan_results.empty,

            "backtest":
                not self.trade_log.empty,

            "reports":
                len(self.report_outputs) > 0,

        }

    ###########################################################################
    # Backtest Stage
    ###########################################################################

    def run_backtest(
        self,
        dataframe: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Execute backtesting stage.
        """

        logger.info(
            "Running backtest stage."
        )

        if dataframe is None:

            dataframe = self.scan_results


        if dataframe.empty:

            raise ValueError(
                "Scanner results are empty. "
                "Run scanner stage first."
            )


        self.trade_log = (

            self.backtest.run_backtest(

                dataframe,

            )

        )


        self.equity_curve = (

            self.backtest.equity_curve

        )


        self.statistics = (

            self.backtest.statistics.copy()

        )


        logger.info(

            "Backtest completed: %d trades.",

            len(self.trade_log),

        )


        return self.trade_log


    ###########################################################################
    # Reporting Stage
    ###########################################################################

    def run_reports(
        self,
    ) -> dict[str, pd.DataFrame]:
        """
        Execute reporting stage.
        """

        logger.info(
            "Running reporting stage."
        )


        if self.scan_results.empty:

            raise ValueError(
                "Scanner results unavailable."
            )


        self.reports.prepare_data(

            scan_results=self.scan_results,

            trade_log=self.trade_log,

            equity_curve=self.equity_curve,

            statistics=self.statistics,

        )


        self.report_outputs = (

            self.reports.run_reports()

        )


        logger.info(

            "Reports generated: %d.",

            len(self.report_outputs),

        )


        return self.report_outputs


    ###########################################################################
    # Export Results
    ###########################################################################

    def results(
        self,
    ) -> PipelineResult:
        """
        Return complete pipeline results.
        """

        return PipelineResult(

            metadata={

                "pipeline_run_id":
                    self.pipeline_run_id,

                "started":
                    self.pipeline_started,

                "finished":
                    self.pipeline_finished,

                "execution_seconds":
                    self.execution_time(),

            },

            data={

                "market_data":
                    self.market_data,

                "indicator_data":
                    self.indicator_data,

                "scan_results":
                    self.scan_results,

                "trade_log":
                    self.trade_log,

                "equity_curve":
                    self.equity_curve,

            },

            statistics=self.statistics,

            reports=self.report_outputs,

            status=(
                "completed"
                if self.is_complete
                else "incomplete"
            ),

            execution_time=self.execution_time(),

            errors=[],

            warnings=[],

        )

    ###########################################################################
    # Validation
    ###########################################################################

    def validate_pipeline(
        self,
    ) -> None:
        """
        Validate pipeline execution state.
        """

        logger.info(
            "Validating pipeline state."
        )


        stages = self.stage_status()


        failed = [

            name

            for name, status in stages.items()

            if not status

        ]


        if failed:

            logger.warning(

                "Incomplete stages: %s",

                failed,

            )

    ###########################################################################
    # Master Pipeline Runner
    ###########################################################################

    def run(
        self,
        *,
        pipeline_run_id: str | None = None,
        period: str = "10y",
        interval: str = "1d",
    ) -> dict[str, Any]:
        """
        Execute complete Swing Scanner pipeline.

        Pipeline
        --------

            Market Data

                ↓

            Indicators

                ↓

            Scanner

                ↓

            Backtest

                ↓

            Reports
        """

        self.reset()

        self.pipeline_started = datetime.now(UTC)

        if pipeline_run_id is None:

            pipeline_run_id = (

                self.pipeline_started

                .strftime(

                    "%Y%m%d_%H%M%S"

                )

            )


        logger.info(
            "Starting pipeline: %s",
            pipeline_run_id,
        )


        try:

            ###################################################################
            # Market Data
            ###################################################################

            self.update_progress(
                "Downloading market data...",
                0.10,
            )

            logger.info("[1/5] Market data stage started")

            self.failed_stage = "market_data"
            
            self.run_market_data(

                pipeline_run_id=pipeline_run_id,

                period=period,

                interval=interval,

            )


            ###################################################################
            # Indicators
            ###################################################################

            self.update_progress(
                "Calculating indicators...",
                0.40,
            )

            logger.info("[2/5] Market data stage started")
            
            self.failed_stage = "indicators"
            
            self.run_indicators()


            ###################################################################
            # Scanner
            ###################################################################

            self.update_progress(
                "Running swing scanner...",
                0.60,
            )

            logger.info("[3/5] Market data stage started")
            
            self.failed_stage = "scanner"
            
            self.run_scanner()


            ###################################################################
            # Backtest
            ###################################################################

            self.update_progress(
                "Running backtest...",
                0.75,
            )

            logger.info("[4/5] Market data stage started")
            
            self.failed_stage = "backtest"
             
            self.run_backtest()


            ###################################################################
            # Reports
            ###################################################################

            self.update_progress(
                "Generating reports...",
                0.90,
            )

            logger.info("[5/5] Market data stage started")
            
            self.failed_stage = "reports"
            
            self.run_reports()

            self.pipeline_finished = datetime.now(UTC)

            self.update_progress(
                "Pipeline completed successfully.",
                1.0,
            )            


            logger.info(

                "Pipeline completed successfully."
            )

            return self.results()


        except Exception as exc:

            logger.exception(

                "Pipeline execution failed: %s",
                self.failed_stage,
                exc,

            )

            raise


    ###########################################################################
    # Execution Metadata
    ###########################################################################

    def execution_time(
        self,
    ) -> float:
        """
        Return pipeline execution time in seconds.
        """

        if (

            self.pipeline_started is None

            or self.pipeline_finished is None

        ):

            return 0.0


        return (

            self.pipeline_finished

            - self.pipeline_started

        ).total_seconds()


    ###########################################################################
    # Health Check
    ###########################################################################

    def health_check(
        self,
    ) -> dict[str, Any]:
        """
        Return pipeline health status.
        """

        stages = self.stage_status()

        return {

            "status":

                "healthy"

                if all(stages.values())

                else "incomplete",


            "stages":

                stages,


            "execution_seconds":

                self.execution_time(),


            "completed":

                self.is_complete,

        }


    ###########################################################################
    # Final Summary
    ###########################################################################

    def summary(
        self,
    ) -> dict[str, Any]:
        """
        Return complete pipeline summary.
        """

        return {

            "pipeline":
                "MarketPipeline",

            "market_rows":
                len(self.market_data),

            "indicator_rows":
                len(self.indicator_data),

            "scan_results":
                len(self.scan_results),

            "trades":
                len(self.trade_log),

            "equity_points":
                len(self.equity_curve),

            "reports":
                len(self.report_outputs),

            "execution_seconds":
                self.execution_time(),

            "completed":
                self.is_complete,

            "started":
                self.pipeline_started,

            "finished":
                self.pipeline_finished,

        }


###############################################################################
# Self Test
###############################################################################

if __name__ == "__main__":

    pipeline = MarketPipeline()

    print(
        "Pipeline initialized"
    )

    result = pipeline.run(
        pipeline_run_id="CLI_TEST",
        period="1y",
        interval="1d",
    )

    print(
        "\nPipeline Completed"
    )

    print(
        result.summary()
    )