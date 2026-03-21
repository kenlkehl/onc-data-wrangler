"""Data exploration API router."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/data", tags=["data"])


def _scan_files(path: Path) -> list[Path]:
    """Find CSV and parquet files in a path."""
    if path.is_file():
        return [path] if path.suffix in (".csv", ".parquet") else []
    if path.is_dir():
        files = sorted(path.glob("*.csv")) + sorted(path.glob("*.parquet"))
        # Also check subdirectories one level deep
        for sub in sorted(path.iterdir()):
            if sub.is_dir():
                files.extend(sorted(sub.glob("*.csv")))
                files.extend(sorted(sub.glob("*.parquet")))
        return files
    return []


@router.get("/files")
async def list_files(paths: str) -> JSONResponse:
    """List data files with metadata."""
    file_list: list[dict[str, Any]] = []

    for path_str in paths.split(","):
        p = Path(path_str.strip())
        if not p.exists():
            continue

        for f in _scan_files(p):
            info: dict[str, Any] = {
                "path": str(f),
                "name": f.name,
                "size_bytes": f.stat().st_size,
            }
            try:
                if f.suffix == ".parquet":
                    import pyarrow.parquet as pq

                    schema = pq.read_schema(f)
                    info["type"] = "parquet"
                    info["columns"] = [
                        {"name": field.name, "type": str(field.type)}
                        for field in schema
                    ]
                    info["row_count"] = pq.read_metadata(f).num_rows
                else:
                    import pandas as pd

                    df = pd.read_csv(f, nrows=0, low_memory=False)
                    info["type"] = "csv"
                    info["columns"] = [
                        {"name": col, "type": "unknown"} for col in df.columns
                    ]
                    info["row_count"] = None
            except Exception as exc:
                logger.warning("Failed to read metadata for %s: %s", f, exc)
                info["type"] = f.suffix.lstrip(".")
                info["columns"] = []
                info["row_count"] = None

            file_list.append(info)

    return JSONResponse(file_list)


@router.get("/preview")
async def preview_file(path: str, limit: int = 50, offset: int = 0) -> JSONResponse:
    """Preview rows from a data file."""
    resolved = Path(path)
    if not resolved.exists():
        raise HTTPException(404, f"File not found: {path}")

    try:
        if resolved.suffix == ".parquet":
            import duckdb

            con = duckdb.connect()
            df = con.execute(
                "SELECT * FROM read_parquet(?) LIMIT ? OFFSET ?",
                [str(resolved), limit, offset],
            ).fetchdf()
            total = con.execute(
                "SELECT COUNT(*) FROM read_parquet(?)", [str(resolved)]
            ).fetchone()[0]
            con.close()
        else:
            import pandas as pd

            # Count total rows (fast line count)
            total = sum(1 for _ in open(resolved)) - 1
            df = pd.read_csv(
                resolved,
                skiprows=range(1, offset + 1) if offset > 0 else None,
                nrows=limit,
                low_memory=False,
            )

        columns = list(df.columns)
        rows = df.where(df.notna(), None).values.tolist()

        return JSONResponse(
            {"columns": columns, "rows": rows, "total_rows": total}
        )
    except Exception as exc:
        raise HTTPException(400, f"Failed to preview file: {exc}")


@router.get("/column-stats")
async def column_stats(path: str, column: str) -> JSONResponse:
    """Get aggregate statistics for a column."""
    resolved = Path(path)
    if not resolved.exists():
        raise HTTPException(404, f"File not found: {path}")

    try:
        import pandas as pd

        if resolved.suffix == ".parquet":
            df = pd.read_parquet(resolved, columns=[column])
        else:
            df = pd.read_csv(resolved, usecols=[column], low_memory=False)

        series = df[column]
        result: dict[str, Any] = {
            "column": column,
            "dtype": str(series.dtype),
            "non_null_count": int(series.notna().sum()),
            "unique_count": int(series.nunique()),
        }

        if pd.api.types.is_numeric_dtype(series):
            result["numeric_stats"] = {
                "min": float(series.min()) if series.notna().any() else None,
                "max": float(series.max()) if series.notna().any() else None,
                "mean": float(series.mean()) if series.notna().any() else None,
                "median": float(series.median()) if series.notna().any() else None,
            }
        else:
            top = series.value_counts().head(10)
            result["top_values"] = [
                {"value": str(val), "count": int(cnt)}
                for val, cnt in top.items()
            ]

        return JSONResponse(result)
    except Exception as exc:
        raise HTTPException(400, f"Failed to compute column stats: {exc}")


@router.get("/browse")
async def browse_directory(path: str = "") -> JSONResponse:
    """Browse filesystem directories and data files."""
    if not path:
        path = str(Path.home())

    resolved = Path(path).resolve()
    if not resolved.exists():
        raise HTTPException(404, f"Path not found: {path}")
    if not resolved.is_dir():
        # If it's a file, browse its parent
        resolved = resolved.parent

    entries: list[dict[str, Any]] = []

    try:
        for item in sorted(resolved.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
            if item.name.startswith("."):
                continue
            entry: dict[str, Any] = {
                "name": item.name,
                "path": str(item),
                "is_dir": item.is_dir(),
            }
            if item.is_file():
                entry["size_bytes"] = item.stat().st_size
                entry["ext"] = item.suffix.lstrip(".")
            entries.append(entry)
    except PermissionError:
        raise HTTPException(403, f"Permission denied: {path}")

    return JSONResponse({
        "current_path": str(resolved),
        "parent": str(resolved.parent) if resolved != resolved.parent else None,
        "entries": entries,
    })


@router.post("/mkdir")
async def make_directory(body: dict) -> JSONResponse:
    """Create a new directory."""
    path = body.get("path", "")
    if not path:
        raise HTTPException(400, "path is required")

    resolved = Path(path).resolve()
    if resolved.exists():
        raise HTTPException(409, f"Already exists: {path}")

    try:
        resolved.mkdir(parents=False, exist_ok=False)
    except FileNotFoundError:
        raise HTTPException(404, f"Parent directory does not exist: {resolved.parent}")
    except PermissionError:
        raise HTTPException(403, f"Permission denied: {path}")

    return JSONResponse({"path": str(resolved)})


@router.get("/outputs")
async def list_outputs(output_dir: str) -> JSONResponse:
    """List pipeline output artifacts."""
    out = Path(output_dir)
    if not out.exists():
        return JSONResponse({"files": []})

    files: list[dict[str, Any]] = []
    for f in sorted(out.iterdir()):
        if f.is_file() and f.suffix in (
            ".parquet",
            ".csv",
            ".duckdb",
            ".json",
            ".md",
        ):
            files.append(
                {
                    "name": f.name,
                    "path": str(f),
                    "size_bytes": f.stat().st_size,
                    "type": f.suffix.lstrip("."),
                }
            )
    # Check subdirectories
    for subdir in ("extractions", "harmonized"):
        sub = out / subdir
        if sub.is_dir():
            for f in sorted(sub.iterdir()):
                if f.is_file():
                    files.append(
                        {
                            "name": f"{subdir}/{f.name}",
                            "path": str(f),
                            "size_bytes": f.stat().st_size,
                            "type": f.suffix.lstrip("."),
                        }
                    )

    return JSONResponse({"files": files})
