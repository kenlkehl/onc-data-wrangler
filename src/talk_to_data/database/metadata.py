"""Generate schema and summary statistics metadata from a DuckDB database.

Produces markdown files describing table structures and aggregate statistics
for use as MCP resources and LLM context.
"""

import logging
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)


def get_tables(con: duckdb.DuckDBPyConnection) -> list[str]:
    """Get all table names in the database."""
    result = con.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = 'main' ORDER BY table_name"
    ).fetchall()
    return [r[0] for r in result]


def get_columns(
    con, table: str, forbidden_columns: set = None
) -> pd.DataFrame:
    """Get column info for a table, excluding forbidden columns."""
    df = con.execute(
        "SELECT column_name, data_type, is_nullable "
        "FROM information_schema.columns "
        "WHERE table_name = ? AND table_schema = 'main' "
        "ORDER BY ordinal_position",
        [table],
    ).fetchdf()
    if forbidden_columns:
        df = df[~df["column_name"].isin(forbidden_columns)]
    return df


def get_row_count(con, table: str) -> int:
    """Get row count for a table."""
    result = con.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()
    return result[0]


def suppress_count(count: int, min_cell_size: int) -> str:
    """Replace small counts with suppression marker."""
    if count < min_cell_size:
        return f"<{min_cell_size}"
    return str(count)


def generate_schema(
    con,
    project_name: str = "Dataset",
    forbidden_columns: set = None,
) -> str:
    """Generate schema markdown describing all tables."""
    tables = get_tables(con)
    lines = []
    lines.append(f"# {project_name} Schema")
    lines.append("")

    for table in tables:
        row_count = get_row_count(con, table)
        columns = get_columns(con, table, forbidden_columns)

        lines.append(f"## Table: `{table}`")
        lines.append("")
        lines.append(f"- **Rows**: {row_count}")
        lines.append(f"- **Columns**: {len(columns)}")
        lines.append("")
        lines.append("| Column | Type | Nullable |")
        lines.append("|--------|------|----------|")

        for _, row in columns.iterrows():
            col_name = row["column_name"]
            data_type = row["data_type"]
            nullable = row["is_nullable"]
            lines.append(f"| `{col_name}` | {data_type} | {nullable} |")

        lines.append("")

    return "\n".join(lines)


def generate_summary(
    con,
    project_name: str = "Dataset",
    forbidden_columns: set = None,
    min_cell_size: int = 10,
) -> str:
    """Generate summary statistics markdown with cell suppression."""
    tables = get_tables(con)
    lines = []
    lines.append(f"# {project_name} Summary Statistics")
    lines.append("")

    for table in tables:
        row_count = get_row_count(con, table)
        columns = get_columns(con, table, forbidden_columns)

        lines.append(f"## Table: `{table}` ({row_count} rows)")
        lines.append("")

        for _, row in columns.iterrows():
            col_name = row["column_name"]
            data_type = row["data_type"]

            if data_type == "VARCHAR":
                _summarize_categorical(
                    con, table, col_name, min_cell_size, lines
                )
            elif data_type in ("DOUBLE", "BIGINT", "INTEGER", "FLOAT"):
                _summarize_numeric(con, table, col_name, lines)

        lines.append("")

    return "\n".join(lines)


def _summarize_categorical(
    con, table: str, col_name: str, min_cell_size: int, lines: list
):
    """Add categorical column summary to lines."""
    lines.append(f"### `{col_name}` (categorical)")
    lines.append("")

    result = con.execute(
        f'SELECT "{col_name}" AS value, COUNT(*) AS count '
        f'FROM "{table}" '
        f'WHERE "{col_name}" IS NOT NULL '
        f'GROUP BY "{col_name}" '
        f"ORDER BY count DESC "
        f"LIMIT 15"
    ).fetchdf()

    if result.empty:
        lines.append("No non-null values.")
        lines.append("")
        return

    lines.append("| Value | Count |")
    lines.append("|-------|-------|")

    for _, row in result.iterrows():
        value = row["value"]
        count = int(row["count"])
        count_str = suppress_count(count, min_cell_size)
        lines.append(f"| {value} | {count_str} |")

    lines.append("")


def _summarize_numeric(con, table: str, col_name: str, lines: list):
    """Add numeric column summary to lines."""
    lines.append(f"### `{col_name}` (numeric)")
    lines.append("")

    result = con.execute(
        f"SELECT "
        f'COUNT("{col_name}") AS count, '
        f'MIN("{col_name}") AS min, '
        f'MAX("{col_name}") AS max, '
        f'AVG("{col_name}") AS avg, '
        f'MEDIAN("{col_name}") AS median, '
        f"PERCENTILE_CONT({0.25}) WITHIN GROUP "
        f'(ORDER BY "{col_name}") AS q1, '
        f"PERCENTILE_CONT({0.75}) WITHIN GROUP "
        f'(ORDER BY "{col_name}") AS q3 '
        f'FROM "{table}" '
        f'WHERE "{col_name}" IS NOT NULL'
    ).fetchone()

    count, min_val, max_val, avg_val, median_val, q1_val, q3_val = result

    lines.append(f"- **Count**: {count}")
    lines.append(f"- **Min**: {min_val:.2f}")
    lines.append(f"- **Max**: {max_val:.2f}")
    lines.append(f"- **Mean**: {avg_val:.2f}")
    lines.append(f"- **Median**: {median_val:.2f}")
    lines.append(f"- **Q1 (25%)**: {q1_val:.2f}")
    lines.append(f"- **Q3 (75%)**: {q3_val:.2f}")
    lines.append("")
