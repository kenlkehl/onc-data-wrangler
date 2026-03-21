"""
NAACCR Ontology Implementation

Implements OntologyBase for NAACCR v26 cancer registry standards.
Loads field definitions from the authoritative NAACCR data dictionary CSVs
(DataItems.csv, CodeList.csv, AlternateNames.csv) rather than hand-crafted JSON.

Supports site-specific data items (SSDIs) for 22+ cancer types via SchemaRegistry.
Provides a NAACCRCodeResolver for 6-tier code resolution.
"""

import logging
from typing import Dict, List, Optional, Any

from ...base import OntologyBase, DataItem, DataCategory

from .dictionary import NAACCRDictionary, NAACCRDataItem
from .code_resolver import NAACCRCodeResolver
from .schema_registry import SchemaRegistry

logger = logging.getLogger(__name__)


SCHEMA_DISPLAY_NAMES = {
    'lung': 'Lung Cancer', 'lung_v9': 'Lung Cancer (2025+)',
    'breast': 'Breast Cancer', 'prostate': 'Prostate Cancer',
    'colorectal': 'Colorectal Cancer', 'colon_rectum': 'Colorectal Cancer',
    'melanoma': 'Melanoma (Cutaneous)', 'melanoma_skin': 'Melanoma (Cutaneous)',
    'kidney': 'Kidney/Renal Cell Cancer', 'kidney_renal_pelvis': 'Kidney/Renal Cell Cancer',
    'bladder': 'Bladder Cancer', 'pancreas': 'Pancreatic Cancer',
    'head_neck': 'Head and Neck Cancer', 'thyroid': 'Thyroid Cancer',
    'liver': 'Hepatocellular Carcinoma',
    'esophagus': 'Esophageal Cancer', 'stomach': 'Gastric Cancer',
    'ovary': 'Ovarian Cancer', 'uterus': 'Uterine/Endometrial Cancer',
    'cervix': 'Cervical Cancer', 'testis': 'Testicular Cancer',
    'brain': 'Brain/CNS Tumors', 'brain_cns': 'Brain/CNS Tumors',
    'lymphoma': 'Lymphoma', 'leukemia': 'Leukemia',
    'myeloma': 'Multiple Myeloma',
    'generic': 'Other/Unspecified Cancer',
}

# Section names in the NAACCR dictionary that map to base categories
_BASE_SECTIONS = {
    'demographics': [
        'Record ID', 'Patient Set--Demographics',
    ],
    'tumor_identification': [
        'Cancer Identification',
    ],
    'staging': [
        'Stage/Prognostic Factors',
    ],
    'treatment': [
        'Treatment-1st Course', 'Treatment-Subsequent',
    ],
    'followup': [
        'Follow-up/Recurrence/Death',
    ],
}


def _dict_item_to_data_item(item: NAACCRDataItem, codes: dict[str, str] | None = None) -> DataItem:
    """Convert a NAACCRDataItem to the base DataItem format."""
    valid_values = codes if codes else None
    json_field = f"naaccr_{item.item_number}_{item.xml_id}" if item.xml_id else f"naaccr_{item.item_number}"

    return DataItem(
        id=str(item.item_number),
        name=item.name,
        description=item.description[:500] if item.description else "",
        data_type=item.data_type.lower() if item.data_type else "string",
        valid_values=valid_values,
        extraction_hints=item.alternate_names[:5] if item.alternate_names else [],
        required=bool(item.npcr_collect.startswith("R") or item.seer_collect.startswith("R")),
        json_field=json_field,
        naaccr_item=str(item.item_number),
        human_readable_field=item.xml_id,
    )


class NAACCROntology(OntologyBase):
    """
    NAACCR v26 Ontology Implementation.

    Loads from authoritative NAACCR CSV data dictionary files.
    Provides cancer registry data items with site-specific support
    for 22+ cancer types.
    """

    def __init__(self):
        self._dictionary = NAACCRDictionary()
        self._dictionary.load()
        self._code_resolver = NAACCRCodeResolver(self._dictionary)
        self._schema_registry = SchemaRegistry()
        self._items_cache: dict[int, DataItem] = {}

    # ------------------------------------------------------------------
    # Public properties for Extractor access
    # ------------------------------------------------------------------

    @property
    def dictionary(self) -> NAACCRDictionary:
        return self._dictionary

    @property
    def code_resolver(self) -> NAACCRCodeResolver:
        return self._code_resolver

    @property
    def schema_registry(self) -> SchemaRegistry:
        return self._schema_registry

    # ------------------------------------------------------------------
    # OntologyBase required properties
    # ------------------------------------------------------------------

    @property
    def ontology_id(self) -> str:
        return 'naaccr'

    @property
    def display_name(self) -> str:
        return 'NAACCR v26 Cancer Registry'

    @property
    def version(self) -> str:
        return '26.0'

    @property
    def description(self) -> str:
        return 'North American cancer registry fields with site-specific items (dictionary-driven)'

    # ------------------------------------------------------------------
    # OntologyBase required methods
    # ------------------------------------------------------------------

    def get_base_items(self) -> List[DataCategory]:
        """Build base DataCategory objects from the CSV dictionary."""
        categories = []

        for cat_id, section_names in _BASE_SECTIONS.items():
            items: list[DataItem] = []
            for section_name in section_names:
                for dict_item in self._dictionary.get_items_by_section(section_name):
                    if dict_item.year_retired:
                        continue
                    di = self._get_or_create_data_item(dict_item)
                    items.append(di)

            if items:
                categories.append(DataCategory(
                    id=cat_id,
                    name=cat_id.replace('_', ' ').title(),
                    description=f"NAACCR {cat_id.replace('_', ' ')} data items",
                    items=items,
                    per_diagnosis=(cat_id != 'demographics'),
                ))

        return categories

    def get_site_specific_items(self, cancer_type: str) -> List[DataCategory]:
        """Get site-specific data items from the SchemaRegistry."""
        # Map ontology cancer types to schema registry names
        schema_map = {
            'colorectal': 'colon_rectum',
            'melanoma': 'melanoma_skin',
            'kidney': 'kidney_renal_pelvis',
            'brain': 'brain_cns',
        }
        schema = schema_map.get(cancer_type, cancer_type)

        ssdis = self._schema_registry.get_required_ssdis(schema)
        if not ssdis:
            return []

        items: list[DataItem] = []
        for item_num in ssdis:
            dict_item = self._dictionary.get_item(item_num)
            if dict_item is None or dict_item.year_retired:
                continue
            di = self._get_or_create_data_item(dict_item)
            items.append(di)

        if not items:
            return []

        schema_name = SCHEMA_DISPLAY_NAMES.get(cancer_type, cancer_type)
        return [DataCategory(
            id=f'site_specific_{cancer_type}',
            name=f'Site-Specific Data Items ({schema_name})',
            description=self._schema_registry.get_site_context(schema),
            items=items,
            per_diagnosis=True,
        )]

    def get_empty_summary_template(self) -> Dict[str, Any]:
        return {
            'patient': {
                'sex': None, 'date_of_birth': None,
                'race': None, 'ethnicity': None,
            },
            'diagnoses': [],
        }

    def get_empty_diagnosis_template(self, cancer_type: str = 'generic') -> Dict[str, Any]:
        schema_name = SCHEMA_DISPLAY_NAMES.get(cancer_type, cancer_type)
        return {
            'schema_id': cancer_type,
            'schema_name': schema_name,
            'primary_site': None,
            'histology': None,
            'behavior': None,
            'grade': None,
            'date_of_diagnosis': None,
            'staging': {},
            'treatment': {},
        }

    def format_for_prompt(self, cancer_type: str = "generic") -> str:
        """Format ontology items as text for LLM prompts.

        This delegates to SchemaBuilder in the new extraction pipeline,
        but is kept for backward compatibility with MultiOntologyExtractor.
        """
        from ....extraction.schema_builder import SchemaBuilder
        builder = SchemaBuilder()

        # Collect all items for this cancer type
        all_items = []
        for cat in self.get_base_items():
            for item in cat.items:
                dict_item = self._dictionary.get_item(int(item.naaccr_item)) if item.naaccr_item else None
                if dict_item:
                    all_items.append(dict_item)

        for cat in self.get_site_specific_items(cancer_type):
            for item in cat.items:
                dict_item = self._dictionary.get_item(int(item.naaccr_item)) if item.naaccr_item else None
                if dict_item:
                    all_items.append(dict_item)

        if not all_items:
            return "Extract NAACCR cancer registry data items."

        return builder.build_json_format_instructions(all_items, self._code_resolver)

    # ------------------------------------------------------------------
    # OntologyBase optional hooks
    # ------------------------------------------------------------------

    def get_supported_cancer_types(self) -> List[str]:
        return list(SCHEMA_DISPLAY_NAMES.keys())

    def detect_cancer_type(
        self,
        primary_site: str = None,
        histology: str = None,
        diagnosis_year: int = None,
    ) -> str:
        return self._schema_registry.get_schema_for_site_histology(
            primary_site or "", histology or "", None,
        )

    def get_extraction_context(self) -> str:
        return (
            "NAACCR v26 cancer registry extraction. Use ICD-O-3 topography codes "
            "for primary site (C##.#) and 4-digit morphology codes for histology. "
            "All dates in YYYYMMDD format."
        )

    def get_code_resolver(self):
        """Return the NAACCR CSV-based code resolver."""
        return self._code_resolver

    def get_schema_resolver(self):
        """Return the NAACCR schema registry."""
        return self._schema_registry

    def validate_output(self, output: Dict[str, Any]) -> List[str]:
        """Basic validation using code resolver."""
        errors = []
        if not isinstance(output, dict):
            return ["Output must be a dictionary"]
        for field_name, value in output.items():
            if isinstance(value, dict) and "value" in value:
                val = str(value["value"])
            elif isinstance(value, str):
                val = value
            else:
                continue
            # Check primary site format
            if "primary_site" in field_name.lower() and val:
                import re
                if not re.match(r'^C\d{2}(\.\d)?$', val, re.IGNORECASE):
                    errors.append(f"Primary site '{val}' is not valid ICD-O-3 format (C##.#)")
        return errors

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_or_create_data_item(self, dict_item: NAACCRDataItem) -> DataItem:
        """Get cached DataItem or create from dictionary item."""
        if dict_item.item_number in self._items_cache:
            return self._items_cache[dict_item.item_number]

        # Build valid_values from code list
        codes = self._dictionary.get_codes(dict_item.item_number)
        valid_values = None
        if codes:
            valid_values = {c.code: c.description for c in codes}

        di = _dict_item_to_data_item(dict_item, valid_values)
        self._items_cache[dict_item.item_number] = di
        return di
