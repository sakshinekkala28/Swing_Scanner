"""
workflows/publish_reports.py
============================

Report publishing workflow.

Responsibilities
----------------
* Collect generated reports
* Export reports
* Prepare publish artifacts
"""

from __future__ import annotations

###############################################################################
# Standard Library
###############################################################################
import logging
from datetime import datetime
from pathlib import Path
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
# Publish Reports Workflow
###############################################################################


class PublishReportsWorkflow:
    """
    Report publishing workflow.
    """

    ###########################################################################
    # Initialization
    ###########################################################################

    def __init__(
        self,
        output_dir: str = "data/reports",
    ) -> None:
        """
        Initialize publishing workflow.
        """

        self.output_dir = Path(
            output_dir,
        )

        self.output_dir.mkdir(

            parents=True,

            exist_ok=True,

        )


        self.reports: dict[str, pd.DataFrame] = {}

        self.exports: dict[str, str] = {}

        self.started: datetime | None = None

        self.finished: datetime | None = None


        logger.info(
            "PublishReportsWorkflow initialized."
        )


    ###########################################################################
    # Load Reports
    ###########################################################################

    def load_reports(
        self,
        reports: dict[str, pd.DataFrame],
    ) -> None:
        """
        Load report collection.
        """

        if not reports:

            raise ValueError(
                "No reports available."
            )


        self.reports = reports.copy()


    ###########################################################################
    # CSV Export
    ###########################################################################

    def export_csv(
        self,
    ) -> dict[str, str]:
        """
        Export reports as CSV files.
        """

        logger.info(
            "Exporting CSV reports."
        )


        for name, dataframe in self.reports.items():

            path = (

                self.output_dir

                /

                f"{name}.csv"

            )


            dataframe.to_csv(

                path,

                index=False,

            )


            self.exports[name] = str(path)


        return self.exports


    ###########################################################################
    # Excel Export
    ###########################################################################

    def export_excel(
        self,
        filename: str = "backtest_report.xlsx",
    ) -> str:
        """
        Export reports into Excel workbook.
        """

        logger.info(
            "Exporting Excel report."
        )


        path = (

            self.output_dir

            /

            filename

        )


        with pd.ExcelWriter(
            path,
            engine="openpyxl",
        ) as writer:


            for name, dataframe in self.reports.items():

                dataframe.to_excel(

                    writer,

                    sheet_name=name[:31],

                    index=False,

                )


        self.exports["excel"] = str(path)


        return str(path)


    ###########################################################################
    # Export All
    ###########################################################################

    def export_reports(
        self,
    ) -> dict[str, str]:
        """
        Export all report formats.
        """

        self.export_csv()

        self.export_excel()


        return self.exports

    ###########################################################################
    # Validation
    ###########################################################################

    def validate_reports(
        self,
    ) -> None:
        """
        Validate report collection.
        """

        if not self.reports:

            raise ValueError(
                "No reports loaded."
            )


        for name, dataframe in self.reports.items():

            if dataframe.empty:

                logger.warning(

                    "Empty report detected: %s",

                    name,

                )


    ###########################################################################
    # Publish
    ###########################################################################

    def publish(
        self,
        reports: dict[str, pd.DataFrame] | None = None,
    ) -> dict[str, str]:
        """
        Publish reports.
        """

        self.started = datetime.utcnow()


        try:

            if reports is not None:

                self.load_reports(
                    reports,
                )


            self.validate_reports()


            exports = self.export_reports()


            self.finished = datetime.utcnow()


            return exports


        except Exception as exc:

            logger.exception(

                "Report publishing failed: %s",

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
        Return execution duration.
        """

        if not self.started or not self.finished:

            return 0.0


        return (

            self.finished

            -

            self.started

        ).total_seconds()


    ###########################################################################
    # Run
    ###########################################################################

    def run(
        self,
        reports: dict[str, pd.DataFrame],
    ) -> dict[str, str]:
        """
        Execute complete publishing workflow.
        """

        return self.publish(
            reports,
        )


    ###########################################################################
    # Artifact Summary
    ###########################################################################

    def artifacts(
        self,
    ) -> dict[str, str]:
        """
        Return generated artifacts.
        """

        return self.exports.copy()


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
                "PublishReportsWorkflow",

            "reports":
                len(self.reports),

            "exports":
                len(self.exports),

            "output_dir":
                str(self.output_dir),

            "execution_seconds":
                self.execution_time(),

            "started":
                self.started,

            "finished":
                self.finished,

        }
