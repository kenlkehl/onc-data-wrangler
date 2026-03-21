"""Load and index the MSK-CHORD CDM data dictionary from CSV files.

Parses the CDM-Codebook metadata CSV to build a queryable index of all
fields in the MSK-CHORD clinical data model.  Each field is represented
as an ``MSKChordField`` dataclass that satisfies the ``DictionaryItemLike``
protocol so it can be consumed by the extraction pipeline.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

MODULE_DIR = Path(__file__).parent
DEFAULT_DATA_DIR = MODULE_DIR / "data"

# CSV column names (from CDM-Codebook - metadata.csv header row)
_COL_FORM_NAME = "form_name"
_COL_FIELD_NAME = "field_name"
_COL_FIELD_TYPE = "field_type"
_COL_FIELD_LABEL = "field_label"
_COL_FIELD_NOTE = "field_note"
_COL_VALIDATION = "text_validation_type_or_sh"
_COL_NLP_DERIVED = "nlp_derived"
_COL_IDENTIFIER = "identifier"
_COL_REDCAP_INSTANCE = "redcap_repeat_instance"

# CSV column names (from CDM-Codebook - tables.csv header row)
_COL_TABLE_FORM_NAME = "form_name"
_COL_TABLE_SOURCE = "cdm_source_table"
_COL_TABLE_DESC = "table_description"
_COL_TABLE_META_PROJECT = "meta_project_name"

# Map CDM field_type strings to normalised data-type labels used by the
# DictionaryItemLike protocol and the rest of the extraction pipeline.
_TYPE_MAP = {
    "STRING": "string",
    "INT": "integer",
    "DATE": "date",
    "FLOAT": "float",
    "TEXT": "text",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MSKChordField:
    """A single field from the MSK-CHORD CDM data dictionary.

    Satisfies the ``DictionaryItemLike`` protocol.
    """

    field_name: str
    label: str
    note: str
    field_type: str
    validation: str
    table_name: str
    instance_number: int = 0
    is_nlp_derived: bool = False
    is_identifier: bool = False

    # -- DictionaryItemLike protocol properties ----------------------------

    @property
    def field_id(self) -> str:
        """Unique identifier: table_name/field_name."""
        return f"{self.table_name}/{self.field_name}"

    @property
    def name(self) -> str:
        """Human-readable name (field_label from the CSV)."""
        return self.label

    @property
    def prompt_field_name(self) -> str:
        """Field name to use in LLM prompts and JSON output."""
        return self.field_name

    @property
    def length(self) -> int:
        """Maximum character length (0 = unlimited)."""
        return 0

    @property
    def data_type(self) -> str:
        """Normalised data type string."""
        # Check validation rule first (date_mdy -> date)
        if self.validation and "date" in self.validation.lower():
            return "date"
        return _TYPE_MAP.get(self.field_type.upper(), "string") if self.field_type else "string"

    @property
    def description(self) -> str:
        """Description of this field (field_note from the CSV)."""
        return self.note

    @property
    def allowable_values(self) -> str:
        """Free-text description of allowable values, if any.

        For most MSK-CHORD fields this is empty because valid values are
        not enumerated in the codebook CSV.  Specific fields that have
        known value sets (e.g. flags, categorical fields) are derived
        from the field_note when possible.
        """
        return self._infer_allowable_values()

    def _infer_allowable_values(self) -> str:
        """Best-effort extraction of value constraints from the note."""
        note_lower = self.note.lower() if self.note else ""

        # Boolean / indicator fields
        if any(kw in note_lower for kw in ("true/false", "logical (true/false)", "indicator")):
            return "True, False"

        # Yes/No fields
        if any(kw in note_lower for kw in ("(yes/no)", "yes/no")):
            return "Yes, No"

        # Known categorical patterns from the codebook
        if "inpatient or outpatient" in note_lower:
            return "INPATIENT, OUTPATIENT"

        if "stage 1-3" in note_lower and "stage 4" in note_lower:
            return "Stage 1-3, Stage 4"

        if "current/former" in note_lower and "never" in note_lower:
            return "Current/Former, Never, Unknown"

        if "'progressing or mixed'" in note_lower or "'improving or stable'" in note_lower:
            return "Progressing or Mixed, Improving or Stable"

        return ""


@dataclass
class MSKChordTable:
    """Metadata about a CDM table / form."""

    form_name: str
    source_table: str
    description: str
    meta_project: str


# ---------------------------------------------------------------------------
# Dictionary loader
# ---------------------------------------------------------------------------

class MSKChordDictionary:
    """In-memory index of the MSK-CHORD CDM data dictionary.

    Usage::

        d = MSKChordDictionary()
        d.load()
        fields = d.get_fields_by_table("Demographics")
    """

    def __init__(
        self,
        data_dir: Optional[str | Path] = None,
    ) -> None:
        self._data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR

        # Primary indexes
        self._fields: list[MSKChordField] = []
        self._fields_by_table: dict[str, list[MSKChordField]] = {}
        self._fields_by_name: dict[str, MSKChordField] = {}
        self._fields_by_id: dict[str, MSKChordField] = {}
        self._tables: dict[str, MSKChordTable] = {}

        self._loaded = False

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Parse the CDM-Codebook CSV files and build look-up indexes."""
        self._load_tables()
        self._load_metadata()
        self._loaded = True
        logger.info(
            "MSK-CHORD dictionary loaded: %d fields across %d tables",
            len(self._fields),
            len(self._fields_by_table),
        )

    def _load_tables(self) -> None:
        """Load table-level metadata from CDM-Codebook - tables.csv."""
        path = self._data_dir / "CDM-Codebook - tables.csv"
        if not path.exists():
            logger.warning("Tables CSV not found at %s; skipping table metadata", path)
            return

        with open(path, newline="", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                form_name = (row.get(_COL_TABLE_FORM_NAME) or "").strip()
                if not form_name:
                    continue
                # Only store if we haven't seen this form_name yet
                # (the CSV has multiple entries per record_id/form combo)
                if form_name not in self._tables:
                    self._tables[form_name] = MSKChordTable(
                        form_name=form_name,
                        source_table=(row.get(_COL_TABLE_SOURCE) or "").strip(),
                        description=(row.get(_COL_TABLE_DESC) or "").strip(),
                        meta_project=(row.get(_COL_TABLE_META_PROJECT) or "").strip(),
                    )

    def _load_metadata(self) -> None:
        """Load field-level metadata from CDM-Codebook - metadata.csv."""
        path = self._data_dir / "CDM-Codebook - metadata.csv"
        with open(path, newline="", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                field_name = (row.get(_COL_FIELD_NAME) or "").strip()
                form_name = (row.get(_COL_FORM_NAME) or "").strip()

                # Skip header/separator rows that have no field_name or form_name
                if not field_name or not form_name:
                    continue

                # Parse instance number for ordering
                raw_instance = (row.get(_COL_REDCAP_INSTANCE) or "").strip()
                instance_num = _safe_int(raw_instance)

                nlp_flag = (row.get(_COL_NLP_DERIVED) or "").strip().lower()
                id_flag = (row.get(_COL_IDENTIFIER) or "").strip().lower()

                item = MSKChordField(
                    field_name=field_name,
                    label=(row.get(_COL_FIELD_LABEL) or "").strip(),
                    note=(row.get(_COL_FIELD_NOTE) or "").strip(),
                    field_type=(row.get(_COL_FIELD_TYPE) or "").strip(),
                    validation=(row.get(_COL_VALIDATION) or "").strip(),
                    table_name=form_name,
                    instance_number=instance_num,
                    is_nlp_derived=(nlp_flag == "y"),
                    is_identifier=(id_flag == "y"),
                )

                self._fields.append(item)
                self._fields_by_table.setdefault(form_name, []).append(item)
                # Index by bare field_name (last-wins for duplicates across tables)
                self._fields_by_name[field_name] = item
                # Index by composite field_id
                self._fields_by_id[item.field_id] = item

    # ------------------------------------------------------------------
    # Public query API
    # ------------------------------------------------------------------

    def get_all_fields(self) -> list[MSKChordField]:
        """Return all fields in the dictionary."""
        return list(self._fields)

    def get_fields_by_table(self, table_name: str) -> list[MSKChordField]:
        """Return all fields belonging to *table_name* (form_name)."""
        return list(self._fields_by_table.get(table_name, []))

    def get_field(self, field_name: str) -> Optional[MSKChordField]:
        """Look up a single field by its bare field_name."""
        return self._fields_by_name.get(field_name)

    def get_field_by_id(self, field_id: str) -> Optional[MSKChordField]:
        """Look up a field by its composite id (table_name/field_name)."""
        return self._fields_by_id.get(field_id)

    def get_table_info(self, table_name: str) -> Optional[MSKChordTable]:
        """Return table-level metadata, or None."""
        return self._tables.get(table_name)

    def get_nlp_derived_fields(self) -> list[MSKChordField]:
        """Return all NLP-derived fields."""
        return [f for f in self._fields if f.is_nlp_derived]

    def get_identifier_fields(self) -> list[MSKChordField]:
        """Return all PHI/identifier fields."""
        return [f for f in self._fields if f.is_identifier]

    def get_fields_with_allowable_values(self) -> list[MSKChordField]:
        """Return fields that have inferred allowable values."""
        return [f for f in self._fields if f.allowable_values]

    @property
    def all_table_names(self) -> list[str]:
        """Return all table/form names that have at least one field."""
        return sorted(self._fields_by_table.keys())

    @property
    def loaded(self) -> bool:
        return self._loaded


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_int(value: str, default: int = 0) -> int:
    try:
        return int(value)
    except (ValueError, TypeError):
        return default
