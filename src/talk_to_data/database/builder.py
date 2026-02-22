"""Build a DuckDB database from extracted and harmonized data.

Creates tables for each data category (diagnosis, biomarker, etc.)
plus a cohort table. Applies de-identification and filtering.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd

from ..config import ProjectConfig

logger = logging.getLogger(__name__)


def _load_cohort_ids(output_dir: str) -> list | None:
    """Load original cohort patient IDs, or None if not available."""
    path = Path(output_dir) / "cohort_ids.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def filter_columns_by_non_missing(
    df: pd.DataFrame, min_non_missing: int = 10
) -> pd.DataFrame:
    """Remove columns where fewer than min_non_missing values are non-null."""
    cols_to_keep = [
        col for col in df.columns if df[col].notna().sum() >= min_non_missing
    ]
    dropped = set(df.columns) - set(cols_to_keep)
    if dropped:
        logger.info("Dropped columns with fewer than %d non-missing values: %s",
                     min_non_missing, sorted(dropped))
    return df[cols_to_keep]


def _deidentify_ids(
    df: pd.DataFrame,
    id_column: str,
    prefix: str,
    id_map: dict = None,
) -> pd.DataFrame:
    """Replace real IDs with sequential anonymized IDs.

    If id_map is None, builds a fresh mapping from IDs in this DataFrame.
    """
    df = df.copy()
    if id_map is None:
        unique_ids = sorted(df[id_column].dropna().unique())
        id_map = {
            old_id: f"{prefix}_{i:06d}"
            for i, old_id in enumerate(unique_ids, start=1)
        }
    df[id_column] = df[id_column].map(id_map)
    return df


def _sanitize_table_name(name: str) -> str:
    """Convert a category name to a valid SQL table name."""
    name = name.lower().replace(" ", "_").replace("-", "_")
    name = "".join(c if c.isalnum() or c == "_" else "_" for c in name)
    if not name:
        return "unknown"
    if name[0].isdigit():
        name = "t_" + name
    return name


def _table_exists(con, table_name: str) -> bool:
    """Check if a table already exists in the database."""
    result = con.execute(
        "SELECT COUNT(*) FROM information_schema.tables "
        "WHERE table_name = ? AND table_schema = 'main'",
        [table_name],
    ).fetchone()
    return result[0] > 0


class DatabaseBuilder:
    """Build a DuckDB database from extracted and harmonized data.

    Creates tables for each data category (diagnosis, biomarker, etc.)
    plus a cohort table. Applies de-identification and filtering.
    """

    def __init__(self, config: ProjectConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.db_path = config.db_path
        self.record_id_prefix = config.database.record_id_prefix
        self.min_non_missing = config.database.min_non_missing
        self.deidentify_dates = config.database.deidentify_dates
        self._birth_dates = None

    def build(self) -> Path:
        """Build the full database. Returns Path to the created DuckDB file."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        if self.db_path.exists():
            logger.info("Removing existing database: %s", self.db_path)
            self.db_path.unlink()

        id_map = self._collect_all_patient_ids()

        if self.deidentify_dates:
            self._birth_dates = self._load_birth_dates(id_map)

        con = duckdb.connect(str(self.db_path))
        try:
            self._load_cohort(con, id_map)
            self._load_extractions(con, id_map)
            self._load_harmonized(con, id_map)
            self._log_summary(con)
        finally:
            con.close()

        logger.info("Database built: %s", self.db_path)
        return self.db_path

    def _collect_all_patient_ids(self) -> dict[str, str]:
        """Build a consistent original-ID → de-identified-ID mapping.

        If the cohort stage was run, its id_map is reconstructed from
        cohort_ids.json (the original IDs saved by the pipeline).  This
        ensures extraction and harmonized data use the same de-identified
        IDs as the cohort table.

        If no cohort_ids.json exists (cohort stage was skipped), IDs are
        collected from extraction shards and harmonized files and a fresh
        mapping is created.
        """
        cohort_ids = _load_cohort_ids(str(self.output_dir))
        if cohort_ids:
            # Reconstruct the same mapping the CohortBuilder used
            sorted_ids = sorted(set(str(x) for x in cohort_ids), key=str)
            id_map = {
                old_id: f"{self.record_id_prefix}_{i:06d}"
                for i, old_id in enumerate(sorted_ids, start=1)
            }
            logger.info("Reconstructed cohort id_map: %d patients", len(id_map))
            return id_map

        # Fallback: no cohort stage — collect from extraction/harmonized data
        all_ids = set()

        extractions_dir = self.output_dir / "extractions"
        if extractions_dir.exists():
            for shard in sorted(extractions_dir.glob("shard_*.parquet")):
                df = pd.read_parquet(shard)
                if "patient_id" in df.columns:
                    all_ids.update(df["patient_id"].dropna().unique())

        harmonized_dir = self.output_dir / "harmonized"
        if harmonized_dir.exists():
            for parquet_file in sorted(harmonized_dir.glob("*.parquet")):
                df = pd.read_parquet(parquet_file)
                if "record_id" in df.columns:
                    all_ids.update(df["record_id"].dropna().unique())

        sorted_ids = sorted(str(i) for i in all_ids)
        id_map = {
            old_id: f"{self.record_id_prefix}_{i:06d}"
            for i, old_id in enumerate(sorted_ids, start=1)
        }
        logger.info("Collected %d unique patient IDs (no cohort)", len(id_map))
        return id_map

    def _load_birth_dates(self, id_map: dict) -> Optional[dict]:
        """Load birth dates from cohort for date de-identification.

        Returns a mapping from de-identified record_id to birth_date.
        The cohort file already has de-identified record_ids and birth_date,
        so we can build the mapping directly.
        """
        cohort_parquet = self.output_dir / "cohort.parquet"
        cohort_csv = self.output_dir / "cohort.csv"

        if cohort_parquet.exists():
            df = pd.read_parquet(cohort_parquet)
        elif cohort_csv.exists():
            df = pd.read_csv(cohort_csv)
        else:
            logger.warning("No cohort file found for birth date loading")
            return None

        if "record_id" not in df.columns or "birth_date" not in df.columns:
            logger.warning(
                "Cohort file missing 'record_id' or 'birth_date' columns"
            )
            return None

        df["birth_date"] = pd.to_datetime(df["birth_date"], errors="coerce")

        # Cohort file already contains de-identified record_ids
        birth_dates = {}
        for _, row in df.iterrows():
            record_id = str(row["record_id"])
            if pd.notna(row["birth_date"]):
                birth_dates[record_id] = row["birth_date"]

        logger.info("Loaded %d birth dates for date de-identification",
                     len(birth_dates))
        return birth_dates

    def _deidentify_dates_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert date columns to years_since_birth and calendar_year.

        For each date column: adds {col}_years_since_birth (float),
        {col}_calendar_year (Int64). Drops the original date column.
        Requires record_id column and pre-loaded birth dates.
        """
        if self._birth_dates is None or "record_id" not in df.columns:
            return df

        skip_columns = {"record_id", "category", "source", "birth_date"}
        date_cols = []

        for col in df.columns:
            if col in skip_columns:
                continue
            if col.endswith("_years_since_birth") or col.endswith("_calendar_year"):
                continue

            if pd.api.types.is_datetime64_any_dtype(df[col]):
                date_cols.append(col)
            elif df[col].dtype == object:
                # Check if >50% of non-null values parse as dates
                non_null = df[col].dropna()
                if len(non_null) > 0:
                    parsed = pd.to_datetime(non_null, errors="coerce")
                    if parsed.notna().sum() / len(non_null) > 0.5:
                        df[col] = pd.to_datetime(df[col], errors="coerce")
                        date_cols.append(col)

        for col in date_cols:
            birth_series = df["record_id"].map(self._birth_dates)
            years_since = (df[col] - birth_series).dt.days / 365.25
            df[f"{col}_years_since_birth"] = years_since
            df[f"{col}_calendar_year"] = df[col].dt.year.astype("Int64")
            df = df.drop(columns=[col])

        return df

    def _load_cohort(self, con, id_map: dict):
        """Load the cohort table from output directory.

        The cohort file is already de-identified by CohortBuilder, so
        record_id values are not re-mapped here.
        """
        cohort_parquet = self.output_dir / "cohort.parquet"
        cohort_csv = self.output_dir / "cohort.csv"

        if cohort_parquet.exists():
            df = pd.read_parquet(cohort_parquet)
        elif cohort_csv.exists():
            df = pd.read_csv(cohort_csv)
        else:
            logger.warning("No cohort file found, skipping cohort table")
            return

        # Cohort is already de-identified by CohortBuilder — no re-mapping.

        # Drop birth_date from the final cohort table — it was kept in the
        # parquet only so _load_birth_dates can use it for de-identifying
        # dates in extraction/harmonized tables.
        if "birth_date" in df.columns:
            df = df.drop(columns=["birth_date"])

        df = filter_columns_by_non_missing(df, self.min_non_missing)

        con.execute("CREATE TABLE cohort AS SELECT * FROM df")
        logger.info("Loaded cohort table: %d rows, %d columns",
                     len(df), len(df.columns))

    def _load_extractions(self, con, id_map: dict):
        """Load extraction shard parquets into per-category tables."""
        extractions_dir = self.output_dir / "extractions"
        if not extractions_dir.exists():
            logger.warning("No extractions directory found, skipping")
            return

        shards = sorted(extractions_dir.glob("shard_*.parquet"))
        if not shards:
            logger.warning("No extraction shards found, skipping")
            return

        dfs = []
        for shard in shards:
            df = pd.read_parquet(shard)
            dfs.append(df)

        df = pd.concat(dfs, ignore_index=True)

        if "patient_id" in df.columns:
            df = df.rename(columns={"patient_id": "record_id"})

        if "record_id" in df.columns:
            df = _deidentify_ids(df, "record_id", self.record_id_prefix, id_map)

        if self.deidentify_dates:
            df = self._deidentify_dates_df(df)

        # Create per-category tables
        if "category" in df.columns:
            for category, group_df in df.groupby("category"):
                table_name = _sanitize_table_name(str(category))
                if _table_exists(con, table_name):
                    con.execute(
                        f'INSERT INTO "{table_name}" SELECT * FROM group_df'
                    )
                else:
                    con.execute(
                        f'CREATE TABLE "{table_name}" AS SELECT * FROM group_df'
                    )
                logger.info("Loaded extraction category '%s' -> table '%s': %d rows",
                            category, table_name, len(group_df))

        # Also create bulk extractions table
        con.execute("CREATE TABLE extractions AS SELECT * FROM df")
        logger.info("Loaded bulk extractions table: %d rows", len(df))

    def _load_harmonized(self, con, id_map: dict):
        """Load harmonized structured data into tables."""
        harmonized_dir = self.output_dir / "harmonized"
        if not harmonized_dir.exists():
            logger.warning("No harmonized directory found, skipping")
            return

        parquet_files = sorted(harmonized_dir.glob("*.parquet"))
        if not parquet_files:
            logger.warning("No harmonized parquet files found, skipping")
            return

        for parquet_file in parquet_files:
            df = pd.read_parquet(parquet_file)

            if "record_id" in df.columns:
                df = _deidentify_ids(
                    df, "record_id", self.record_id_prefix, id_map
                )

            if self.deidentify_dates:
                df = self._deidentify_dates_df(df)

            df = filter_columns_by_non_missing(df, self.min_non_missing)

            # Table name from file stem: rsplit on '_', take from index 1 to -1
            stem = parquet_file.stem
            parts = stem.rsplit("_", 1)
            table_name = _sanitize_table_name(parts[0] if len(parts) > 1 else stem)

            if _table_exists(con, table_name):
                con.execute(
                    f'INSERT INTO "{table_name}" SELECT * FROM df'
                )
            else:
                con.execute(
                    f'CREATE TABLE "{table_name}" AS SELECT * FROM df'
                )
            logger.info("Loaded harmonized file '%s' -> table '%s': %d rows, %d columns",
                        parquet_file.name, table_name, len(df), len(df.columns))

    def _log_summary(self, con):
        """Log a summary of all tables."""
        tables = con.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main' ORDER BY table_name"
        ).fetchall()

        logger.info("Database summary:")
        for (table_name,) in tables:
            row_count = con.execute(
                f'SELECT COUNT(*) FROM "{table_name}"'
            ).fetchone()[0]
            col_count = con.execute(
                "SELECT COUNT(*) FROM information_schema.columns "
                "WHERE table_name = ? AND table_schema = 'main'",
                [table_name],
            ).fetchone()[0]
            logger.info("  Table '%s': %d rows, %d columns",
                        table_name, row_count, col_count)
