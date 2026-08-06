"""
engine/storage.py
=================

Enterprise Storage Manager for the Swing Scanner platform.

Responsibilities
----------------
* Centralized persistence layer
* DuckDB connection management
* Filesystem management
* Parquet storage
* CSV storage
* Excel exports
* Archive & cleanup
* Storage health monitoring

Architecture
------------
Every module must use StorageManager.

    Scanner
        │
        ▼
 StorageManager
        │
 ┌──────┼──────────┐
 ▼      ▼          ▼
DuckDB Parquet    Excel
        │
        ▼
 Streamlit

Notes
-----
This is the ONLY module allowed to interact directly with
DuckDB and filesystem persistence.
"""

from __future__ import annotations

###############################################################################
# Standard Library
###############################################################################
import logging
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from shutil import copy2, copytree, rmtree

###############################################################################
# Third Party
###############################################################################
import duckdb
import pandas as pd

###############################################################################
# Local Imports
###############################################################################
from config import settings

###############################################################################
# Logger
###############################################################################

logger = logging.getLogger(__name__)

###############################################################################
# Storage Manager
###############################################################################


class StorageManager:
    """
    Central storage manager.

    Handles all persistent storage operations for the application.

    Storage Backends
    ----------------
    • DuckDB
    • Parquet
    • CSV
    • Excel

    This class is intentionally the only persistence layer.
    Business modules should never access files or databases directly.
    """

    ###########################################################################
    # Construction
    ###########################################################################

    def __init__(self) -> None:
        """Initialize storage manager."""

        self.paths = settings.paths
        self.storage = settings.storage

        #######################################################################
        # Project Directories
        #######################################################################

        self.root_path: Path = self.paths.root

        self.data_path: Path = self.paths.data

        self.raw_path: Path = self.paths.raw

        self.cache_path: Path = self.paths.cache

        self.export_path: Path = self.paths.exports

        self.report_path: Path = self.paths.reports

        #######################################################################
        # Storage
        #######################################################################

        self.duckdb_path: Path = self.storage.duckdb

        self.parquet_path: Path = self.storage.parquet

        self.excel_path: Path = self.storage.excel

        #######################################################################

        self.initialize()

    ###########################################################################
    # Initialization
    ###########################################################################

    def initialize(self) -> None:
        """
        Initialize project storage.

        Creates all required directories.

        Safe to call multiple times.
        """

        directories = (

            self.data_path,

            self.raw_path,

            self.cache_path,

            self.export_path,

            self.report_path,

            self.parquet_path,

            self.duckdb_path.parent,

        )

        for directory in directories:

            directory.mkdir(
                parents=True,
                exist_ok=True,
            )

        logger.info(
            "Storage initialized successfully."
        )

    ###########################################################################
    # DuckDB Connection
    ###########################################################################

    @contextmanager
    def connection(
        self,
    ) -> Generator[duckdb.DuckDBPyConnection, None, None]:
        """
        Return a managed DuckDB connection.

        Connection Lifecycle
        --------------------

        Open

            ↓

        Execute

            ↓

        Commit

            ↓

        Close
        """

        logger.debug(
            "Opening DuckDB connection."
        )

        connection = duckdb.connect(
            str(self.duckdb_path)
        )

        try:

            yield connection

            connection.commit()

        except Exception:

            connection.rollback()

            logger.exception(
                "DuckDB transaction failed."
            )

            raise

        finally:

            connection.close()

            logger.debug(
                "DuckDB connection closed."
            )

    ###########################################################################
    # Validation
    ###########################################################################

    @staticmethod
    def validate_path(
        path: Path,
    ) -> Path:
        """
        Validate filesystem path.
        """

        if not isinstance(path, Path):

            raise TypeError(
                "Expected pathlib.Path."
            )

        return path

    @staticmethod
    def validate_dataframe(
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Validate dataframe.
        """

        if not isinstance(
            dataframe,
            pd.DataFrame,
        ):

            raise TypeError(
                "Expected pandas DataFrame."
            )

        return dataframe

    @staticmethod
    def validate_dataset(
        dataset: str,
    ) -> str:
        """
        Validate dataset name.
        """

        if not dataset:

            raise ValueError(
                "Dataset cannot be empty."
            )

        dataset = dataset.strip()

        invalid = (
            "..",
            "/",
            "\\",
            ":",
        )

        if any(
            token in dataset
            for token in invalid
        ):

            raise ValueError(
                f"Invalid dataset name: {dataset}"
            )

        return dataset

    @staticmethod
    def exists(
        path: Path,
    ) -> bool:
        """
        Check whether a path exists.
        """

        return Path(path).exists()

    ###########################################################################
    # Internal Helpers
    ###########################################################################

    def _ensure_directory(
        self,
        directory: Path,
    ) -> Path:
        """
        Ensure directory exists.
        """

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        return directory

    def _timestamp(
        self,
    ) -> str:
        """
        UTC timestamp.

        Returns
        -------
        str
        """

        return (
            pd.Timestamp.utcnow()
            .strftime("%Y%m%d_%H%M%S")
        )

###############################################################################
# DuckDB Operations
###############################################################################

def execute(
    self,
    sql: str,
    parameters: tuple | list | None = None,
) -> None:
    """
    Execute a SQL statement.

    Parameters
    ----------
    sql
        SQL statement.

    parameters
        Optional SQL parameters.
    """

    logger.debug("Executing SQL:\n%s", sql)

    with self.connection() as conn:

        if parameters:

            conn.execute(sql, parameters)

        else:

            conn.execute(sql)


def query(
    self,
    sql: str,
    parameters: tuple | list | None = None,
) -> pd.DataFrame:
    """
    Execute SQL query.

    Returns
    -------
    pandas.DataFrame
    """

    logger.debug("Executing query.")

    with self.connection() as conn:

        if parameters:

            return conn.execute(
                sql,
                parameters,
            ).fetchdf()

        return conn.execute(sql).fetchdf()


###############################################################################
# Table Operations
###############################################################################

@staticmethod
def _validate_table_name(
    table: str,
) -> str:
    """
    Validate SQL table name.
    """

    if not table:

        raise ValueError(
            "Table name cannot be empty."
        )

    table = table.strip()

    invalid = (
        "..",
        "/",
        "\\",
        ";",
        "--",
        " ",
    )

    if any(
        token in table
        for token in invalid
    ):

        raise ValueError(
            f"Invalid table name: {table}"
        )

    return table


def table_exists(
    self,
    table: str,
) -> bool:
    """
    Check whether table exists.
    """

    table = self._validate_table_name(table)

    sql = """
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_schema='main'
        AND table_name=?
    """

    result = self.query(
        sql,
        (table,),
    )

    return bool(
        result.iloc[0, 0]
    )


def list_tables(
    self,
) -> list[str]:
    """
    Return database tables.
    """

    sql = """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema='main'
        ORDER BY table_name
    """

    tables = self.query(sql)

    return tables[
        "table_name"
    ].tolist()


def read_table(
    self,
    table: str,
) -> pd.DataFrame:
    """
    Read entire table.
    """

    table = self._validate_table_name(table)

    if not self.table_exists(table):

        raise ValueError(
            f"Table '{table}' not found."
        )

    return self.query(
        f'SELECT * FROM "{table}"'
    )


def write_table(
    self,
    dataframe: pd.DataFrame,
    table: str,
    *,
    mode: str = "replace",
    pipeline_run_id: str,
    source: str,
) -> None:
    """
    Write dataframe to DuckDB.

    Modes
    -----
    replace
    append
    fail
    """

    dataframe = self.validate_dataframe(
        dataframe
    )

    table = self._validate_table_name(
        table
    )

    if dataframe.empty:

        logger.warning(
            "Empty dataframe skipped."
        )

        return

    df = dataframe.copy()

    if "created_at" not in df:

        df["created_at"] = (
            pd.Timestamp.utcnow()
        )

    if "pipeline_run_id" not in df:

        df[
            "pipeline_run_id"
        ] = pipeline_run_id

    if "source" not in df:

        df["source"] = source

    with self.connection() as conn:

        conn.register(
            "__dataframe__",
            df,
        )

        exists = conn.execute(
            """
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_schema='main'
            AND table_name=?
            """,
            (table,),
        ).fetchone()[0] > 0

        if mode == "replace":

            conn.execute(
                f"""
                CREATE OR REPLACE TABLE "{table}" AS
                SELECT *
                FROM __dataframe__
                """
            )

        elif mode == "append":

            if exists:

                conn.execute(
                    f"""
                    INSERT INTO "{table}"
                    SELECT *
                    FROM __dataframe__
                    """
                )

            else:

                conn.execute(
                    f"""
                    CREATE TABLE "{table}" AS
                    SELECT *
                    FROM __dataframe__
                    """
                )

        elif mode == "fail":

            if exists:

                raise ValueError(
                    f"Table '{table}' already exists."
                )

            conn.execute(
                f"""
                CREATE TABLE "{table}" AS
                SELECT *
                FROM __dataframe__
                """
            )

        else:

            raise ValueError(
                "Mode must be "
                "'replace', 'append', or 'fail'."
            )

        conn.unregister(
            "__dataframe__"
        )

    logger.info(
        "Table '%s' written (%d rows).",
        table,
        len(df),
    )


def drop_table(
    self,
    table: str,
) -> None:
    """
    Drop table.
    """

    table = self._validate_table_name(
        table
    )

    self.execute(
        f'DROP TABLE IF EXISTS "{table}"'
    )


def truncate_table(
    self,
    table: str,
) -> None:
    """
    Remove all rows.
    """

    table = self._validate_table_name(
        table
    )

    if not self.table_exists(table):

        return

    self.execute(
        f'DELETE FROM "{table}"'
    )


###############################################################################
# Database Maintenance
###############################################################################

def vacuum(
    self,
) -> None:
    """
    Optimize DuckDB database.
    """

    self.execute("VACUUM")


def analyze(
    self,
) -> None:
    """
    Refresh optimizer statistics.
    """

    self.execute("ANALYZE")


###############################################################################
# Dataset Helpers
###############################################################################

def _dataset_directory(
    self,
    dataset: str,
) -> Path:
    """
    Return dataset directory.

    Example
    -------
    data/cache/market_data/
    """

    dataset = self.validate_dataset(dataset)

    directory = (
        self.parquet_path
        / dataset
    )

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return directory


def _latest_parquet(
    self,
    dataset: str,
) -> Path:
    """
    Return latest parquet file.
    """

    return (
        self._dataset_directory(dataset)
        / "latest.parquet"
    )


###############################################################################
# Parquet
###############################################################################

def save_parquet(
    self,
    dataframe: pd.DataFrame,
    dataset: str,
    *,
    snapshot: bool = True,
    compression: str = "snappy",
) -> Path:
    """
    Save dataframe as parquet.

    Parameters
    ----------
    dataframe
        DataFrame.

    dataset
        Dataset name.

    snapshot
        Save timestamped snapshot.

    compression
        Parquet compression.
    """

    dataframe = self.validate_dataframe(
        dataframe
    )

    dataset = self.validate_dataset(
        dataset
    )

    directory = self._dataset_directory(
        dataset
    )

    latest = (
        directory
        / "latest.parquet"
    )

    dataframe.to_parquet(
        latest,
        index=False,
        compression=compression,
    )

    if snapshot:

        snapshot_file = (
            directory
            / f"{self._timestamp()}.parquet"
        )

        dataframe.to_parquet(
            snapshot_file,
            index=False,
            compression=compression,
        )

    logger.info(
        "Saved parquet dataset '%s'.",
        dataset,
    )

    return latest


def load_parquet(
    self,
    dataset: str,
) -> pd.DataFrame:
    """
    Load latest parquet dataset.
    """

    dataset = self.validate_dataset(
        dataset
    )

    path = self._latest_parquet(
        dataset
    )

    if not path.exists():

        raise FileNotFoundError(path)

    logger.info(
        "Loading parquet '%s'.",
        dataset,
    )

    return pd.read_parquet(path)


###############################################################################
# CSV
###############################################################################

def save_csv(
    self,
    dataframe: pd.DataFrame,
    dataset: str,
    *,
    encoding: str = "utf-8",
) -> Path:
    """
    Save dataframe as CSV.
    """

    dataframe = self.validate_dataframe(
        dataframe
    )

    dataset = self.validate_dataset(
        dataset
    )

    path = (
        self.export_path
        / f"{dataset}.csv"
    )

    dataframe.to_csv(
        path,
        index=False,
        encoding=encoding,
    )

    logger.info(
        "Saved CSV '%s'.",
        dataset,
    )

    return path


def load_csv(
    self,
    dataset: str,
    *,
    encoding: str = "utf-8",
) -> pd.DataFrame:
    """
    Load CSV dataset.
    """

    dataset = self.validate_dataset(
        dataset
    )

    path = (
        self.export_path
        / f"{dataset}.csv"
    )

    if not path.exists():

        raise FileNotFoundError(path)

    logger.info(
        "Loading CSV '%s'.",
        dataset,
    )

    return pd.read_csv(
        path,
        encoding=encoding,
    )


###############################################################################
# Excel
###############################################################################

def export_excel(
    self,
    dataframe: pd.DataFrame,
    report: str,
    *,
    sheet_name: str = "Sheet1",
    overwrite: bool = True,
    index: bool = False,
) -> Path:
    """
    Export dataframe to Excel.
    """

    dataframe = self.validate_dataframe(
        dataframe
    )

    report = self.validate_dataset(
        report
    )

    path = (
        self.report_path
        / f"{report}.xlsx"
    )

    if path.exists() and not overwrite:

        raise FileExistsError(path)

    with pd.ExcelWriter(
        path,
        engine="openpyxl",
    ) as writer:

        dataframe.to_excel(
            writer,
            sheet_name=sheet_name,
            index=index,
        )

        worksheet = writer.sheets[
            sheet_name
        ]

        worksheet.freeze_panes = "A2"

        for column in worksheet.columns:

            width = max(
                len(str(cell.value))
                if cell.value is not None
                else 0
                for cell in column
            )

            worksheet.column_dimensions[
                column[0].column_letter
            ].width = min(
                max(width + 2, 12),
                40,
            )

    logger.info(
        "Excel exported '%s'.",
        report,
    )

    return path


def export_sheet(
    self,
    dataframe: pd.DataFrame,
    report: str,
    sheet_name: str,
    *,
    index: bool = False,
) -> Path:
    """
    Add or replace worksheet.
    """

    dataframe = self.validate_dataframe(
        dataframe
    )

    report = self.validate_dataset(
        report
    )

    path = (
        self.report_path
        / f"{report}.xlsx"
    )

    mode = "a" if path.exists() else "w"

    writer_kwargs = {
        "engine": "openpyxl",
        "mode": mode,
    }

    if mode == "a":
        writer_kwargs["if_sheet_exists"] = "replace"

    with pd.ExcelWriter(
        path,
        **writer_kwargs,
    ) as writer:

        dataframe.to_excel(
            writer,
            sheet_name=sheet_name,
            index=index,
        )

        worksheet = writer.sheets[
            sheet_name
        ]

        worksheet.freeze_panes = "A2"

    logger.info(
        "Worksheet '%s' exported.",
        sheet_name,
    )

    return path

###############################################################################
# Archive
###############################################################################

def archive_reports(
    self,
    *,
    archive_name: str | None = None,
) -> Path:
    """
    Archive report directory.

    Returns
    -------
    Path
    """

    timestamp = archive_name or self._timestamp()

    archive = (
        self.report_path
        / "archive"
        / timestamp
    )

    archive.mkdir(
        parents=True,
        exist_ok=True,
    )

    for file in self.report_path.glob("*"):

        if (
            file.is_file()
            and file.suffix.lower()
            in {
                ".xlsx",
                ".csv",
                ".html",
                ".pdf",
            }
        ):

            copy2(
                file,
                archive / file.name,
            )

    logger.info(
        "Reports archived -> %s",
        archive,
    )

    return archive


def archive_cache(
    self,
    *,
    archive_name: str | None = None,
) -> Path:
    """
    Archive cache.

    Returns
    -------
    Path
    """

    timestamp = archive_name or self._timestamp()

    archive = (
        self.cache_path
        / "archive"
        / timestamp
    )

    archive.mkdir(
        parents=True,
        exist_ok=True,
    )

    for item in self.cache_path.iterdir():

        if item.name == "archive":
            continue

        destination = archive / item.name

        if item.is_dir():

            copytree(
                item,
                destination,
                dirs_exist_ok=True,
            )

        else:

            copy2(
                item,
                destination,
            )

    logger.info(
        "Cache archived -> %s",
        archive,
    )

    return archive


###############################################################################
# Cleanup
###############################################################################

def cleanup_cache(
    self,
) -> None:
    """
    Remove cache except archive.
    """

    for item in self.cache_path.iterdir():

        if item.name == "archive":
            continue

        if item.is_dir():

            rmtree(
                item,
                ignore_errors=True,
            )

        else:

            item.unlink(
                missing_ok=True,
            )

    logger.info("Cache cleaned.")


def cleanup_reports(
    self,
) -> None:
    """
    Remove generated reports.

    Archive folders are preserved.
    """

    for file in self.report_path.glob("*"):

        if file.name == "archive":
            continue

        if file.is_file():

            file.unlink(
                missing_ok=True,
            )

    logger.info(
        "Reports cleaned."
    )


###############################################################################
# File Utilities
###############################################################################

@staticmethod
def delete_file(
    path: Path,
) -> None:
    """
    Delete file.
    """

    path.unlink(
        missing_ok=True,
    )


@staticmethod
def delete_directory(
    directory: Path,
) -> None:
    """
    Delete directory.
    """

    if directory.exists():

        rmtree(
            directory,
            ignore_errors=True,
        )


@staticmethod
def file_size(
    path: Path,
) -> int:
    """
    File size in bytes.
    """

    if not path.exists():

        return 0

    return path.stat().st_size


@staticmethod
def list_files(
    directory: Path,
    pattern: str = "*",
) -> list[Path]:
    """
    List files.
    """

    return sorted(
        directory.glob(pattern)
    )


###############################################################################
# Validation
###############################################################################

def database_exists(
    self,
) -> bool:
    """
    Check DuckDB database.
    """

    return self.duckdb_path.exists()


def report_exists(
    self,
    report: str,
) -> bool:
    """
    Check Excel report.
    """

    report = self.validate_dataset(report)

    return (
        self.report_path
        / f"{report}.xlsx"
    ).exists()


def dataset_exists(
    self,
    dataset: str,
) -> bool:
    """
    Check parquet dataset.
    """

    dataset = self.validate_dataset(
        dataset
    )

    return self._latest_parquet(
        dataset
    ).exists()


###############################################################################
# Health
###############################################################################

def health_check(
    self,
) -> dict[str, object]:
    """
    Storage health.

    Returns
    -------
    dict
    """

    return {

        "duckdb": self.database_exists(),

        "raw": self.raw_path.exists(),

        "cache": self.cache_path.exists(),

        "exports": self.export_path.exists(),

        "reports": self.report_path.exists(),

        "datasets": len(
            list(
                self.cache_path.glob("*")
            )
        ),

        "tables": len(
            self.list_tables()
        ),

    }