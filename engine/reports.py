"""
engine/reports.py
=================

Institutional Reporting Engine.

Responsibilities
----------------
* Generate scan reports
* Generate trade reports
* Generate portfolio reports
* Generate dashboard reports
* Export Excel workbooks
* Export summary statistics

This module does NOT:

    * Download market data
    * Calculate indicators
    * Generate trading signals
    * Execute backtests

Those responsibilities belong to:

    data.py
    indicators.py
    scanner.py
    backtest.py
"""

from __future__ import annotations

###############################################################################
# Standard Library
###############################################################################
import logging
from pathlib import Path
from typing import Any

###############################################################################
# Third Party
###############################################################################
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
# Report Engine
###############################################################################


class ReportEngine:
    """
    Institutional Reporting Engine.

    Pipeline
    --------

        Scanner Results

            ↓

        Trade Log

            ↓

        Portfolio Statistics

            ↓

        Equity Curve

            ↓

        Report Builder

            ↓

        Excel

            ↓

        Dashboard

            ↓

        Summary
    """

    ###########################################################################
    # Initialization
    ###########################################################################

    def __init__(
        self,
    ) -> None:
        """
        Initialize reporting engine.
        """

        self.settings = settings

        self.storage = StorageManager()

        self.scan_results = pd.DataFrame()

        self.trade_log = pd.DataFrame()

        self.equity_curve = pd.DataFrame()

        self.statistics: dict[str, Any] = {}

        self.report_directory = (
            self.settings.paths.reports
        )

        logger.info(
            "ReportEngine initialized."
        )

    ###########################################################################
    # Data Preparation
    ###########################################################################

    def prepare_data(
        self,
        scan_results: pd.DataFrame,
        trade_log: pd.DataFrame,
        equity_curve: pd.DataFrame,
        statistics: dict[str, Any],
    ) -> None:
        """
        Prepare report inputs.
        """

        self.scan_results = scan_results.copy()

        self.trade_log = trade_log.copy()

        self.equity_curve = equity_curve.copy()

        self.statistics = statistics.copy()

        logger.info(

            "Prepared reporting datasets."

        )

    ###########################################################################
    # Scan Report
    ###########################################################################

    def scan_report(
        self,
    ) -> pd.DataFrame:
        """
        Build swing scanner report.
        """

        logger.info(
            "Generating scan report."
        )

        if self.scan_results.empty:

            return pd.DataFrame()

        columns = [

            "Symbol",

            "Date",

            "SIGNAL",

            "TOTAL_SCORE",

            "RANK",

            "ENTRY",

            "STOP_LOSS",

            "TARGET",

            "RISK_REWARD",

        ]

        report = self.scan_results.copy()

        columns = [

            column

            for column in columns

            if column in report.columns

        ]

        report = report[columns]

        report = report.sort_values(

            "RANK",

            ascending=True,

        )

        return report.reset_index(
            drop=True,
        )


    ###########################################################################
    # Trade Report
    ###########################################################################

    def trade_report(
        self,
    ) -> pd.DataFrame:
        """
        Build completed trade report.
        """

        logger.info(
            "Generating trade report."
        )

        if self.trade_log.empty:

            return pd.DataFrame()

        columns = [

            "Symbol",

            "ENTRY_DATE",

            "ENTRY_PRICE",

            "EXIT_DATE",

            "EXIT_PRICE",

            "EXIT_REASON",

            "QUANTITY",

            "GROSS_PNL",

            "NET_PNL",

            "RETURN_PCT",

            "HOLDING_DAYS",

        ]

        report = self.trade_log.copy()

        columns = [

            column

            for column in columns

            if column in report.columns

        ]

        report = report[columns]

        return report.reset_index(
            drop=True,
        )


    ###########################################################################
    # Performance Report
    ###########################################################################

    def performance_report(
        self,
    ) -> pd.DataFrame:
        """
        Build performance summary.
        """

        logger.info(
            "Generating performance report."
        )

        if not self.statistics:

            return pd.DataFrame()

        report = pd.DataFrame(

            {

                "Metric": list(

                    self.statistics.keys()

                ),

                "Value": list(

                    self.statistics.values()

                ),

            }

        )

        return report


    ###########################################################################
    # Summary Report
    ###########################################################################

    def summary_report(
        self,
    ) -> pd.DataFrame:
        """
        Generate executive summary.
        """

        logger.info(
            "Generating summary report."
        )

        summary = {

            "Scan Rows": len(
                self.scan_results,
            ),

            "Trades": len(
                self.trade_log,
            ),

            "Equity Points": len(
                self.equity_curve,
            ),

            "Initial Capital": self.statistics.get(
                "INITIAL_CAPITAL",
                0,
            ),

            "Final Capital": self.statistics.get(
                "FINAL_CAPITAL",
                0,
            ),

            "Total Return %": self.statistics.get(
                "TOTAL_RETURN_PCT",
                0,
            ),

            "Win Rate %": self.statistics.get(
                "WIN_RATE",
                0,
            ),

            "Profit Factor": self.statistics.get(
                "PROFIT_FACTOR",
                0,
            ),

            "Max Drawdown %": self.statistics.get(
                "MAX_DRAWDOWN_PCT",
                0,
            ),

        }

        return pd.DataFrame(

            {
                "Metric": summary.keys(),
                "Value": summary.values(),
            }
        )


    ###########################################################################
    # Equity Curve Report
    ###########################################################################

    def equity_report(
        self,
    ) -> pd.DataFrame:
        """
        Generate equity curve report.
        """

        logger.info(
            "Generating equity curve report."
        )

        if self.equity_curve.empty:

            return pd.DataFrame()

        report = self.equity_curve.copy()

        columns = [

            "DATE",

            "EQUITY",

            "RUNNING_MAX",

            "DRAWDOWN",

            "DRAWDOWN_PCT",

        ]

        columns = [

            column

            for column in columns

            if column in report.columns

        ]

        return report[columns].reset_index(
            drop=True,
        )


    ###########################################################################
    # Drawdown Report
    ###########################################################################

    def drawdown_report(
        self,
    ) -> pd.DataFrame:
        """
        Generate drawdown report.
        """

        logger.info(
            "Generating drawdown report."
        )

        if self.equity_curve.empty:

            return pd.DataFrame()

        report = self.equity_curve.copy()

        report["DRAWDOWN_RANK"] = (

            report["DRAWDOWN_PCT"]

            .rank(

                method="dense",

                ascending=True,

            )

        )

        return report.sort_values(

            "DRAWDOWN_PCT"

        ).reset_index(

            drop=True,

        )


    ###########################################################################
    # Monthly Returns Report
    ###########################################################################

    def monthly_returns_report(
        self,
    ) -> pd.DataFrame:
        """
        Generate monthly returns.
        """

        logger.info(
            "Generating monthly returns."
        )

        if self.trade_log.empty:

            return pd.DataFrame()

        report = self.trade_log.copy()

        report["ENTRY_DATE"] = pd.to_datetime(

            report["ENTRY_DATE"]

        )

        report["YEAR"] = (

            report["ENTRY_DATE"]

            .dt.year

        )

        report["MONTH"] = (

            report["ENTRY_DATE"]

            .dt.month_name()

        )

        monthly = (

            report

            .groupby(

                [

                    "YEAR",

                    "MONTH",

                ],

                as_index=False,

            )

            .agg(

                {

                    "NET_PNL": "sum",

                    "RETURN_PCT": "mean",

                }

            )

        )

        return monthly


    ###########################################################################
    # Portfolio Allocation Report
    ###########################################################################

    def portfolio_report(
        self,
    ) -> pd.DataFrame:
        """
        Portfolio allocation summary.
        """

        logger.info(
            "Generating portfolio report."
        )

        if self.trade_log.empty:

            return pd.DataFrame()

        report = self.trade_log.copy()

        if "Sector" in report.columns:

            allocation = (

                report

                .groupby(

                    "Sector",

                    as_index=False,

                )

                .agg(

                    {

                        "QUANTITY": "sum",

                        "NET_PNL": "sum",

                    }

                )

            )

            allocation["WEIGHT_PCT"] = (

                allocation["QUANTITY"]

                /

                allocation["QUANTITY"].sum()

                * 100

            )

            return allocation

        return pd.DataFrame()


    ###########################################################################
    # Dashboard Dataset
    ###########################################################################

    def dashboard(
        self,
    ) -> pd.DataFrame:
        """
        Build Streamlit dashboard dataset.
        """

        logger.info(
            "Generating dashboard dataset."
        )

        dashboard = {

            "Scan Rows": len(

                self.scan_results

            ),

            "Trades": len(

                self.trade_log

            ),

            "Buy Signals": (

                self.scan_results["SIGNAL"]

                .eq("BUY")

                .sum()

                if "SIGNAL"

                in self.scan_results.columns

                else 0

            ),

            "Average Score": (

                self.scan_results["TOTAL_SCORE"]

                .mean()

                if "TOTAL_SCORE"

                in self.scan_results.columns

                else 0

            ),

            "Win Rate": self.statistics.get(

                "WIN_RATE",

                0,

            ),

            "Total Return %": self.statistics.get(

                "TOTAL_RETURN_PCT",

                0,

            ),

            "Profit Factor": self.statistics.get(

                "PROFIT_FACTOR",

                0,

            ),

            "Max Drawdown %": self.statistics.get(

                "MAX_DRAWDOWN_PCT",

                0,

            ),

        }

        return pd.DataFrame(

            {

                "Metric": dashboard.keys(),

                "Value": dashboard.values(),

            }

        )

    ###########################################################################
    # Excel Export
    ###########################################################################

    def export_excel(
        self,
    ) -> Path:
        """
        Export complete Excel workbook.
        """

        logger.info(
            "Exporting Excel workbook."
        )

        output_file = (

            self.report_directory

            / "swing_scanner_report.xlsx"

        )

        with pd.ExcelWriter(

            output_file,

            engine="openpyxl",

        ) as writer:

            self.scan_report().to_excel(

                writer,

                sheet_name="Scanner",

                index=False,

            )

            self.trade_report().to_excel(

                writer,

                sheet_name="Trades",

                index=False,

            )

            self.performance_report().to_excel(

                writer,

                sheet_name="Performance",

                index=False,

            )

            self.summary_report().to_excel(

                writer,

                sheet_name="Summary",

                index=False,

            )

            self.equity_report().to_excel(

                writer,

                sheet_name="Equity Curve",

                index=False,

            )

            self.drawdown_report().to_excel(

                writer,

                sheet_name="Drawdown",

                index=False,

            )

            self.monthly_returns_report().to_excel(

                writer,

                sheet_name="Monthly Returns",

                index=False,

            )

            self.portfolio_report().to_excel(

                writer,

                sheet_name="Portfolio",

                index=False,

            )

            self.dashboard().to_excel(

                writer,

                sheet_name="Dashboard",

                index=False,

            )

        logger.info(

            "Workbook exported: %s",

            output_file,

        )

        return output_file


    ###########################################################################
    # Individual Exports
    ###########################################################################

    def export_scan_report(
        self,
    ) -> None:
        """
        Export scanner report.
        """

        self.storage.save_excel(

            self.scan_report(),

            filename="scan_report.xlsx",

        )


    def export_trade_report(
        self,
    ) -> None:
        """
        Export trade report.
        """

        self.storage.save_excel(

            self.trade_report(),

            filename="trade_report.xlsx",

        )


    def export_performance_report(
        self,
    ) -> None:
        """
        Export performance report.
        """

        self.storage.save_excel(

            self.performance_report(),

            filename="performance_report.xlsx",

        )


    def export_equity_report(
        self,
    ) -> None:
        """
        Export equity report.
        """

        self.storage.save_excel(

            self.equity_report(),

            filename="equity_curve.xlsx",

        )


    def export_dashboard(
        self,
    ) -> None:
        """
        Export dashboard report.
        """

        self.storage.save_excel(

            self.dashboard(),

            filename="dashboard.xlsx",

        )


    ###########################################################################
    # Statistics Export
    ###########################################################################

    def export_statistics(
        self,
    ) -> pd.DataFrame:
        """
        Export statistics table.
        """

        statistics = pd.DataFrame(

            {

                "Metric": self.statistics.keys(),

                "Value": self.statistics.values(),

            }

        )

        self.storage.save_excel(

            statistics,

            filename="statistics.xlsx",

        )

        return statistics


    ###########################################################################
    # Export All
    ###########################################################################

    def export_all(
        self,
    ) -> None:
        """
        Export every report.
        """

        logger.info(
            "Exporting all reports."
        )

        self.export_excel()

        self.export_scan_report()

        self.export_trade_report()

        self.export_performance_report()

        self.export_equity_report()

        self.export_dashboard()

        self.export_statistics()

    ###########################################################################
    # Validation
    ###########################################################################

    def validate(
        self,
    ) -> None:
        """
        Validate report inputs.
        """

        logger.info(
            "Validating reporting inputs."
        )

        if self.scan_results.empty:

            logger.warning(
                "Scan results are empty."
            )

        if self.trade_log.empty:

            logger.warning(
                "Trade log is empty."
            )

        if self.equity_curve.empty:

            logger.warning(
                "Equity curve is empty."
            )

        if not self.statistics:

            logger.warning(
                "Statistics dictionary is empty."
            )


    ###########################################################################
    # Persistence
    ###########################################################################

    def save_reports(
        self,
    ) -> None:
        """
        Persist reports using StorageManager.
        """

        logger.info(
            "Persisting reports."
        )

        reports = {

            "scan_report": self.scan_report(),

            "trade_report": self.trade_report(),

            "performance_report": self.performance_report(),

            "summary_report": self.summary_report(),

            "equity_report": self.equity_report(),

            "drawdown_report": self.drawdown_report(),

            "monthly_returns": self.monthly_returns_report(),

            "portfolio_report": self.portfolio_report(),

            "dashboard": self.dashboard(),

        }

        for name, dataframe in reports.items():

            if dataframe.empty:

                continue

            self.storage.write_table(

                dataframe=dataframe,

                table=name,

                mode="replace",

                source="reports",

            )

            self.storage.save_parquet(

                dataframe,

                dataset=name,

            )


    ###########################################################################
    # Reporting Pipeline
    ###########################################################################

    def run_reports(
        self,
    ) -> dict[str, pd.DataFrame]:
        """
        Execute complete reporting pipeline.
        """

        logger.info(
            "Starting reporting pipeline."
        )

        self.validate()

        outputs = {

            "scan_report": self.scan_report(),

            "trade_report": self.trade_report(),

            "performance_report": self.performance_report(),

            "summary_report": self.summary_report(),

            "equity_report": self.equity_report(),

            "drawdown_report": self.drawdown_report(),

            "monthly_returns": self.monthly_returns_report(),

            "portfolio_report": self.portfolio_report(),

            "dashboard": self.dashboard(),

        }

        self.export_all()

        self.save_reports()

        logger.info(
            "Reporting pipeline completed successfully."
        )

        return outputs


    ###########################################################################
    # Report Inventory
    ###########################################################################

    def available_reports(
        self,
    ) -> list[str]:
        """
        Return available report names.
        """

        return [

            "scan_report",

            "trade_report",

            "performance_report",

            "summary_report",

            "equity_report",

            "drawdown_report",

            "monthly_returns",

            "portfolio_report",

            "dashboard",

        ]


    ###########################################################################
    # Summary
    ###########################################################################

    def summary(
        self,
    ) -> dict[str, Any]:
        """
        Reporting engine summary.
        """

        return {

            "engine": "ReportEngine",

            "scan_rows": len(
                self.scan_results,
            ),

            "trade_rows": len(
                self.trade_log,
            ),

            "equity_points": len(
                self.equity_curve,
            ),

            "statistics": len(
                self.statistics,
            ),

            "available_reports": len(
                self.available_reports(),
            ),

            "report_directory": str(
                self.report_directory,
            ),

        }
