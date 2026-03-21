"""Load and index the PRISSMM/GENIE BPC data dictionary from the Excel file.

Parses the BPC NSCLC v2.0 public variable synopsis spreadsheet and provides
``PRISSMMField`` objects that satisfy the ``DictionaryItemLike`` protocol.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

MODULE_DIR = Path(__file__).parent
DEFAULT_DATA_DIR = MODULE_DIR / "data"
DEFAULT_EXCEL_FILENAME = "bpc_nsclc_v2.0-public_variable_synopsis.xlsx"
SHEET_NAME = "Variable Synopsis"

# Map Excel "Data Type" values to normalised types
_DTYPE_MAP = {
    "character": "string",
    "numeric": "number",
    "date": "date",
    "text": "text",
}

# Regex for stripping bullet-point markers from Values cells
_BULLET_RE = re.compile(r"^\s*[\u2022\u2023\u25E6\u00B7\-\*]\s*", re.MULTILINE)
# Separator injected by openpyxl for Excel in-cell line breaks
_CELL_BREAK = "_x000D_\n"


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class PRISSMMField:
    """A single PRISSMM data-dictionary variable.

    Satisfies the ``DictionaryItemLike`` protocol.
    """

    variable_name: str
    field_label: str
    dataset: str
    data_type: str
    valid_values_raw: str
    valid_values_list: list[str] = field(default_factory=list)

    # -- DictionaryItemLike protocol properties ----------------------------

    @property
    def field_id(self) -> str:
        """Unique identifier: the PRISSMM variable name."""
        return self.variable_name

    @property
    def name(self) -> str:
        return self.field_label

    @property
    def prompt_field_name(self) -> str:
        return self.variable_name

    @property
    def length(self) -> int:
        """PRISSMM does not define fixed lengths; return 0 (unlimited)."""
        return 0

    @property
    def description(self) -> str:
        return self.field_label

    @property
    def allowable_values(self) -> str:
        """Formatted string of allowable values for LLM context."""
        if self.valid_values_list:
            return ", ".join(self.valid_values_list)
        return ""

    @property
    def valid_values_dict(self) -> dict[str, str]:
        """Return {value: value} mapping for code resolution.

        PRISSMM values are self-descriptive labels (not code/description
        pairs), so the key and value are the same string.
        """
        return {v: v for v in self.valid_values_list}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_values_cell(raw: str) -> list[str]:
    """Parse a Values cell into a list of individual value strings.

    The Excel file uses bullet points (``\u2022\\tValue_x000D_\\n``) to
    delimit values within a single cell.
    """
    if not raw or not isinstance(raw, str) or raw.strip().lower() == "nan":
        return []

    # Split on the Excel in-cell line break marker
    parts = raw.split(_CELL_BREAK)
    results: list[str] = []
    for part in parts:
        # Also split on real newlines in case some snuck through
        for subpart in part.split("\n"):
            cleaned = subpart.strip()
            # Strip bullet markers
            cleaned = _BULLET_RE.sub("", cleaned).strip()
            # Strip leading tab (common after bullet)
            cleaned = cleaned.lstrip("\t").strip()
            if cleaned:
                results.append(cleaned)
    return results


def _normalise_dataset_name(raw: str) -> str:
    """Normalise a dataset name to a short snake_case key."""
    raw = raw.strip().lower()
    mapping = {
        "patient-level dataset": "patient",
        "cancer diagnosis dataset": "cancer_diagnosis",
        "cancer-directed regimen dataset": "regimen",
        "prissmm imaging level dataset": "imaging",
        "prissmm pathology level dataset": "pathology",
        "prissmm medical oncologist assessment level dataset": "medical_oncologist_assessment",
        "cancer panel test level dataset": "cancer_panel_test",
    }
    return mapping.get(raw, raw.replace(" ", "_").replace("-", "_"))


# ---------------------------------------------------------------------------
# Dictionary loader
# ---------------------------------------------------------------------------

class PRISSMMDictionary:
    """In-memory index of the PRISSMM/GENIE BPC data dictionary.

    Usage::

        d = PRISSMMDictionary()
        d.load()
        fields = d.get_all_fields()
    """

    def __init__(
        self,
        data_dir: Optional[str | Path] = None,
        excel_filename: str = DEFAULT_EXCEL_FILENAME,
    ) -> None:
        self._data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
        self._excel_filename = excel_filename

        # Primary indexes
        self._fields_by_name: dict[str, PRISSMMField] = {}
        self._fields_by_dataset: dict[str, list[PRISSMMField]] = {}

        self._loaded = False

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Parse the Excel variable synopsis and build look-up indexes."""
        filepath = self._data_dir / self._excel_filename
        if not filepath.exists():
            logger.warning("PRISSMM dictionary file not found: %s", filepath)
            self._loaded = True
            return

        df = pd.read_excel(filepath, sheet_name=SHEET_NAME, engine="openpyxl")

        for _, row in df.iterrows():
            variable_name = str(row.get("Variable Name", "")).strip()
            if not variable_name:
                continue

            raw_values = str(row.get("Values", ""))
            raw_dtype = str(row.get("Data Type", "")).strip().lower()
            dataset_raw = str(row.get("Dataset", "")).strip()
            dataset_key = _normalise_dataset_name(dataset_raw)

            parsed_values = _parse_values_cell(raw_values)
            normalised_dtype = _DTYPE_MAP.get(raw_dtype, raw_dtype or "string")

            fld = PRISSMMField(
                variable_name=variable_name,
                field_label=str(row.get("Field Label", "")).strip(),
                dataset=dataset_key,
                data_type=normalised_dtype,
                valid_values_raw=raw_values if raw_values != "nan" else "",
                valid_values_list=parsed_values,
            )

            self._fields_by_name[variable_name] = fld
            self._fields_by_dataset.setdefault(dataset_key, []).append(fld)

        self._loaded = True
        logger.info(
            "PRISSMM dictionary loaded: %d fields across %d datasets",
            len(self._fields_by_name),
            len(self._fields_by_dataset),
        )

    # ------------------------------------------------------------------
    # Public query API
    # ------------------------------------------------------------------

    def get_all_fields(self) -> list[PRISSMMField]:
        """Return all loaded fields."""
        return list(self._fields_by_name.values())

    def get_fields_by_dataset(self, dataset: str) -> list[PRISSMMField]:
        """Return all fields belonging to *dataset*.

        The *dataset* parameter should be a normalised key such as
        ``'patient'``, ``'cancer_diagnosis'``, ``'regimen'``, etc.
        """
        return list(self._fields_by_dataset.get(dataset, []))

    def get_field(self, variable_name: str) -> Optional[PRISSMMField]:
        """Return a single field by its PRISSMM variable name, or ``None``."""
        return self._fields_by_name.get(variable_name)

    @property
    def datasets(self) -> list[str]:
        """Return all dataset keys present in the dictionary."""
        return sorted(self._fields_by_dataset.keys())

    @property
    def loaded(self) -> bool:
        return self._loaded

    def get_valid_values_map(self) -> dict[str, dict[str, str]]:
        """Build a ``{field_id: {code: description}}`` mapping.

        Suitable for constructing a ``GenericCodeResolver``.
        """
        result: dict[str, dict[str, str]] = {}
        for fld in self._fields_by_name.values():
            vv = fld.valid_values_dict
            if vv:
                result[fld.field_id] = vv
        return result
