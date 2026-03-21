"""Privacy enforcement layer: cell suppression and output sanitization."""
import numpy as np
import pandas as pd


def sanitize_query_output(df: pd.DataFrame, count_columns: list[str], min_cell: int = 10, forbidden_columns: set = None) -> tuple[pd.DataFrame, bool]:
    """Apply privacy suppression to query output.

    Args:
        df: Query result DataFrame.
        count_columns: Column names identified as containing counts.
        min_cell: Minimum cell size (counts below this are suppressed).
        forbidden_columns: Column names to always strip from output.

    Returns:
        (sanitized_df, suppression_applied) tuple.
    """
    if df.empty:
        return df, False

    df = df.copy()

    # Drop forbidden columns (always strip record_id)
    cols_to_drop = [c for c in df.columns if c.lower() in (forbidden_columns or {'record_id'})]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    suppression_applied = False
    all_columns = list(df.columns)

    for count_col in count_columns:
        if count_col not in df.columns:
            continue
        # Identify rows with small counts
        mask = df[count_col].apply(lambda x: _is_small(x, min_cell))
        if mask.any():
            suppression_applied = True
            # Suppress the count column itself
            df[count_col] = df[count_col].astype(object)
            df.loc[mask, count_col] = f"<{min_cell}"
            # Also suppress rate/percentage columns in those same rows
            for col in all_columns:
                if col != count_col and _is_rate_key(col):
                    df[col] = df[col].astype(object)
                    df.loc[mask, col] = "suppressed"

    return df, suppression_applied


def validate_output_size(n_output_rows: int, n_cohort: int, max_output_fraction: float = 0.5) -> None:
    """Guard against queries that effectively return row-level data.

    Raises ValueError if the output has too many rows relative to the cohort.
    """
    if n_cohort > 0 and n_output_rows > n_cohort * max_output_fraction:
        raise ValueError(
            f"Output has {n_output_rows} rows for a cohort of {n_cohort} patients. "
            f"This exceeds the maximum allowed fraction ({max_output_fraction}). "
            "The query may be returning near-individual-level data."
        )


def _is_small(value, min_cell: int) -> bool:
    """Check if a value is a number below the threshold."""
    if isinstance(value, (int, float, np.integer, np.floating)):
        if np.isnan(value):
            return False
        return value < min_cell
    return False


def _is_rate_key(key: str) -> bool:
    """Check if a key name suggests it contains a rate/percentage."""
    return any(term in key.lower() for term in ('pct', 'percent', 'rate', 'prevalence', 'proportion', 'ci_lower', 'ci_upper', 'survival', 'median_os'))
