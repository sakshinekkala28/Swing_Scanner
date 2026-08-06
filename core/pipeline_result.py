"""
core/pipeline_result.py
=======================

Standard result container for Swing Scanner pipeline.

Provides a structured response object
between workflow layer and application layer.
"""

from __future__ import annotations

###############################################################################
# Standard Library
###############################################################################
from dataclasses import dataclass, field
from typing import Any

###############################################################################
# Third Party
###############################################################################
import pandas as pd

###############################################################################
# Pipeline Result
###############################################################################


@dataclass
class PipelineResult:
    """
    Complete output container for MarketPipeline.
    """

    ###########################################################################
    # Metadata
    ###########################################################################

    metadata: dict[str, Any] = field(
        default_factory=dict,
    )

    ###########################################################################
    # Execution
    ###########################################################################

    status: str = "initialized"

    execution_time: float = 0.0

    errors: list[str] = field(
        default_factory=list,
    )

    warnings: list[str] = field(
        default_factory=list,
    )

    ###########################################################################
    # Data Outputs
    ###########################################################################

    data: dict[str, pd.DataFrame] = field(
        default_factory=dict,
    )

    ###########################################################################
    # Statistics
    ###########################################################################

    statistics: dict[str, Any] = field(
        default_factory=dict,
    )

    ###########################################################################
    # Reports
    ###########################################################################

    reports: dict[str, pd.DataFrame] = field(
        default_factory=dict,
    )

    ###########################################################################
    # Data Access
    ###########################################################################

    def get_data(
        self,
        name: str,
    ) -> pd.DataFrame:
        """
        Return dataframe by name.
        """

        return self.data.get(
            name,
            pd.DataFrame(),
        )


    ###########################################################################
    # Report Access
    ###########################################################################

    def get_report(
        self,
        name: str,
    ) -> pd.DataFrame:
        """
        Return report dataframe by name.
        """

        return self.reports.get(
            name,
            pd.DataFrame(),
        )


    ###########################################################################
    # Status
    ###########################################################################

    def is_valid(
        self,
    ) -> bool:
        """
        Validate pipeline result.
        """

        return bool(
            self.data
            or self.statistics
            or self.reports
        )


    ###########################################################################
    # Summary
    ###########################################################################

    def summary(
        self,
    ) -> dict[str, Any]:
        """
        Return result summary.
        """

        return {

            "metadata":
                self.metadata,

            "datasets":
                list(
                    self.data.keys()
                ),

            "statistics":
                list(
                    self.statistics.keys()
                ),

            "reports":
                list(
                    self.reports.keys()
                ),

            "valid":
                self.is_valid(),

            "status":
                self.status,

            "execution_time":
                self.execution_time,

            "errors":
                len(self.errors),

            "warnings":
                len(self.warnings),
        }

###############################################################################
# Self Test
###############################################################################

if __name__ == "__main__":

    result = PipelineResult(

        metadata={
            "pipeline": "Swing Scanner",
            "status": "completed",
        },

        data={

            "market_data": pd.DataFrame(
                {
                    "Symbol": [
                        "RELIANCE",
                    ],
                    "Close": [
                        2500,
                    ],
                }
            )

        },

        statistics={

            "return": 15.5,

            "win_rate": 62.0,

        },

        reports={

            "summary": pd.DataFrame(
                {
                    "Metric": [
                        "Return",
                    ],
                    "Value": [
                        15.5,
                    ],
                }
            )

        },

    )


    print(
        "Valid:",
        result.is_valid(),
    )


    print(
        "\nSummary:"
    )

    print(
        result.summary()
    )


    print(
        "\nMarket Data:"
    )

    print(
        result.get_data(
            "market_data"
        )
    )