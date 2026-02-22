"""
Pan-TOP Ontology Implementation

Implements OntologyBase for Pan-TOP (Pan-Thoracic Oncology Platform) data extraction.
Provides structured extraction of thoracic malignancy data from clinical notes,
following the extraction schema used by the DFCI Pan-TOP cohort study.

Categories:
- cancer_diagnosis
- cancer_biomarker
- cancer_systemic_therapy_regimen
- cancer_surgery
- cancer_radiation_therapy
- cancer_burden
- smoking_history
"""

import json
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Any

from ...base import OntologyBase, DataItem, DataCategory


MODULE_DIR = Path(__file__).parent

EXTRACTION_CATEGORIES = [
    'cancer_diagnosis',
    'cancer_biomarker',
    'cancer_systemic_therapy_regimen',
    'cancer_surgery',
    'cancer_radiation_therapy',
    'cancer_burden',
    'smoking_history',
]

SUPPORTED_CANCER_TYPES = [
    'lung',
    'mesothelioma',
    'thymus',
    'generic',
]

CANCER_TYPE_DISPLAY = {
    'lung': 'Lung Cancer',
    'mesothelioma': 'Mesothelioma',
    'thymus': 'Thymic Malignancy',
    'generic': 'Thoracic Malignancy (Generic)',
}

SITE_TO_TYPE_MAP = {
    # Lung
    'C340': 'lung',
    'C341': 'lung',
    'C342': 'lung',
    'C343': 'lung',
    'C348': 'lung',
    'C349': 'lung',
    'C34': 'lung',
    # Mesothelioma
    'C450': 'mesothelioma',
    'C451': 'mesothelioma',
    'C452': 'mesothelioma',
    'C457': 'mesothelioma',
    'C459': 'mesothelioma',
    'C45': 'mesothelioma',
    # Pericardium (mesothelioma)
    'C384': 'mesothelioma',
    # Thymus
    'C379': 'thymus',
    'C37': 'thymus',
    # Thymus (benign)
    'D150': 'thymus',
    'D151': 'thymus',
    'D152': 'thymus',
    'D157': 'thymus',
    'D159': 'thymus',
    'D15': 'thymus',
}


def _normalize_site_code(primary_site: str) -> str:
    if not primary_site:
        return ''
    return primary_site.upper().replace('.', '').replace(' ', '')


def _load_json_file(filepath: Path) -> Dict:
    if not filepath.exists():
        return {}
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


class PanTOPOntology(OntologyBase):
    """
    Pan-TOP (Pan-Thoracic Oncology Platform) Ontology Implementation.

    Extracts structured data from clinical notes for thoracic malignancy
    patients, covering diagnosis, biomarkers, systemic therapy, surgery,
    radiation, disease burden, and smoking history.
    """

    @property
    def ontology_id(self) -> str:
        return 'pan_top'

    @property
    def display_name(self) -> str:
        return 'Pan-TOP Thoracic Oncology'

    @property
    def version(self) -> str:
        return '1.0.0'

    def get_base_items(self) -> List[DataCategory]:
        """Load base Pan-TOP data items from JSON files."""
        categories = []
        base_dir = MODULE_DIR / 'base'

        for category_name in EXTRACTION_CATEGORIES:
            filepath = base_dir / f'{category_name}.json'
            if not filepath.exists():
                continue

            data = _load_json_file(filepath)
            items = []
            for item_data in data.get('items', []):
                items.append(DataItem(
                    id=item_data.get('json_field', ''),
                    name=item_data.get('name', ''),
                    description=item_data.get('description', ''),
                    data_type=item_data.get('data_type', 'string'),
                    valid_values=None,
                    extraction_hints=item_data.get('extraction_hints', []),
                    json_field=item_data.get('json_field'),
                    repeatable=category_name != 'smoking_history',
                ))

            categories.append(DataCategory(
                id=category_name,
                name=data.get('category', category_name).replace('_', ' ').title(),
                description=data.get('description', ''),
                items=items,
                per_diagnosis=True,
            ))

        return categories

    def get_site_specific_items(self, cancer_type: str) -> List[DataCategory]:
        """Return per-diagnosis categories for the cancer type.

        Pan-TOP is already thoracic-focused, so all categories apply
        regardless of the specific thoracic subtype. The base items
        are the site-specific items.
        """
        return self.get_base_items()

    def get_empty_summary_template(self) -> Dict[str, Any]:
        """Return empty Pan-TOP JSON structure."""
        return {'extractions': []}

    def get_empty_diagnosis_template(self, cancer_type: str = 'generic') -> Dict[str, Any]:
        """Return empty Pan-TOP diagnosis entry structure."""
        return {
            'cancer_diagnosis': {
                'primary_site': None,
                'overall_stage_at_diagnosis': None,
                't_stage_at_diagnosis': None,
                'n_stage_at_diagnosis': None,
                'm_stage_at_diagnosis': None,
                'histology': None,
                'diagnosis_date': None,
            },
            'biomarkers': [],
            'systemic_therapies': [],
            'surgeries': [],
            'radiation_therapies': [],
            'burden_assessments': [],
            'smoking_history': {
                'categorical_smoking_history': None,
                'pack_years': None,
            },
        }

    def format_for_prompt(self, cancer_type: str) -> str:
        """Format Pan-TOP extraction instructions for LLM prompts.

        Reproduces the extraction instructions from extract_patient_chunked.py,
        adapted for the ontology framework.
        """
        type_display = CANCER_TYPE_DISPLAY.get(cancer_type, 'Thoracic Malignancy')

        lines = []
        lines.append(f'Extract Pan-TOP structured data for: {type_display}')
        lines.append(
            'Extract key information from clinical notes, imaging reports, and pathology reports for patients with thoracic malignancies.')

        lines.append('')
        lines.append(
            'The output should be a list of dictionaries. Each dictionary has a single key (the category name) whose value is a dictionary of subfields.')

        lines.append('')

        lines.append('=== CATEGORIES AND SUBFIELDS ===')
        lines.append('')

        lines.append('1. cancer_diagnosis')
        lines.append('   Subfields:')
        lines.append('   - primary_site (eg lung, thymus, mesothelioma)')
        lines.append('   - overall_stage_at_diagnosis (ie I, IIA, IIIB, IV)')
        lines.append('   - t_stage_at_diagnosis (ie T2a, T3b)')
        lines.append('   - n_stage_at_diagnosis (ie N0, N1, N2, N3)')
        lines.append('   - m_stage_at_diagnosis (ie M0, M1a, M1b)')
        lines.append(
            '   - histology (eg non_small_cell_NOS, small_cell, adenocarcinoma, squamous, adenosquamous, large_cell, large_cell_neuroendocrine, typical_carcinoid, atypical_carcinoid, carcinoid_nos, nut_carcinoma, epithelioid, sarcomatoid, thymoma, thymic_carcinoma)')
        lines.append(
            '   - diagnosis_date (as specific as is known; if only year is indicated, list year; if month is indicated also, list month and year)')
        lines.append(
            '   Guidelines: One entry per cancer diagnosis; recurrences or metastases do not count as a new diagnosis. Be very careful not to include non-cancer diagnoses.')

        lines.append('')

        lines.append('2. cancer_biomarker')
        lines.append('   Subfields:')
        lines.append('   - biomarker_type (eg mutation, rearrangement, expression, amplification)')
        lines.append(
            '   - biomarker_tested (eg EGFR, ALK, ROS1, PD-L1, TTF-1, p40, KRAS, HER2, BRAF, RET, MET, NTRK). Note: EGFR refers to EGFR mutation only, not estimated GFR.')
        lines.append(
            '   - biomarker_result (eg positive, negative, 60%, L858R mutant, CD74-ROS1 fusion)')

        lines.append('')

        lines.append('3. cancer_systemic_therapy_regimen')
        lines.append('   Subfields:')
        lines.append('   - regimen_drug_list (a list of individual drug names)')
        lines.append('   - regimen_start_date (as specific as is known)')
        lines.append('   - regimen_end_date (as specific as known)')
        lines.append('   - regimen_toxicity_list (a list of individual toxicities)')
        lines.append('   Guidelines: Only extract regimens actually administered, not discussed options.')
        lines.append('')

        lines.append('4. cancer_surgery')
        lines.append('   Subfields:')
        lines.append(
            '   - surgery_type (eg lobectomy, wedge_resection, segmentectomy, pneumonectomy, thymectomy, pleurectomy_decortication, extrapleural_pneumonectomy)')
        lines.append('   - surgery_site (eg RUL, RML, RLL, LUL, LLL, lingula, left_lung, thymus)')
        lines.append('   - surgery_date (as specific as is known)')
        lines.append('')

        lines.append('5. cancer_radiation_therapy')
        lines.append('   Subfields:')
        lines.append('   - radiation_type (eg long_course, SBRT, SRS)')
        lines.append('   - radiation_start_date (as specific as is known)')
        lines.append('   - radiation_end_date (as specific as is known)')
        lines.append('   - radiation_site (eg RUL, RML, liver, bone, brain)')
        lines.append('')

        lines.append('6. cancer_burden')
        lines.append('   Subfields:')
        lines.append('   - burden_assessor_type (eg radiologist, oncologist)')
        lines.append('   - burden_assessment_date')
        lines.append('   - burden_type (eg response/improvement, progression/worsening, cancer_site). If cancer_site, add another cancer_site subfield to specify the location.')
        lines.append('   Guidelines: Progression/worsening should only be called if the tumor is growing.')
        lines.append('')

        lines.append('7. smoking_history')
        lines.append('   Subfields:')
        lines.append('   - categorical_smoking_history (eg never, former, current)')
        lines.append('   - pack_years (extract whatever was documented)')
        lines.append('')

        lines.append('=== OUTPUT FORMAT ===')
        lines.append("Output a list of dictionaries under the 'extractions' key.")
        lines.append('Each dictionary has a single key (category name) with subfields as value.')
        lines.append('Never invent information not present in the document.')
        lines.append('If information is missing, omit the subfield entirely.')
        lines.append('All dates should be formatted as YYYY-MM-DD when possible.')
        lines.append('')
        lines.append('Example:')
        lines.append('[')
        lines.append('  {"cancer_diagnosis": {"primary_site": "lung", "histology": "adenocarcinoma", "overall_stage_at_diagnosis": "IVA"}},')
        lines.append('  {"cancer_biomarker": {"biomarker_type": "mutation", "biomarker_tested": "EGFR", "biomarker_result": "L858R"}},')
        lines.append('  {"cancer_systemic_therapy_regimen": {"regimen_drug_list": ["carboplatin", "pemetrexed"]}},')
        lines.append('  {"smoking_history": {"categorical_smoking_history": "former", "pack_years": "30"}}')
        lines.append(']')

        return '\n'.join(lines)

    def get_supported_cancer_types(self) -> List[str]:
        """List supported cancer types."""
        return list(SUPPORTED_CANCER_TYPES)

    def detect_cancer_type(self, primary_site: Optional[str] = None,
                           histology: Optional[str] = None,
                           diagnosis_year: Optional[int] = None) -> str:
        """Map primary site to thoracic cancer type."""
        site_code = _normalize_site_code(primary_site or '')

        if site_code in SITE_TO_TYPE_MAP:
            return SITE_TO_TYPE_MAP[site_code]

        for prefix_len in (3, 2):
            if len(site_code) >= prefix_len:
                prefix = site_code[:prefix_len]
                if prefix in SITE_TO_TYPE_MAP:
                    return SITE_TO_TYPE_MAP[prefix]

        return 'generic'

    def get_extraction_context(self) -> str:
        """Additional context for Pan-TOP extraction."""
        return (
            'Pan-TOP (Pan-Thoracic Oncology Platform) extracts structured data '
            'from clinical notes for thoracic malignancy patients at DFCI. '
            'The extraction schema covers diagnosis, biomarkers, systemic therapy, '
            'surgery, radiation, disease burden, and smoking history. '
            'Each extraction entry is a single-key dictionary with the category '
            'name as key and subfields as value. Multiple entries of the same '
            'category are allowed (e.g., multiple biomarkers, multiple regimens).'
        )

    def validate_output(self, output: Dict[str, Any]) -> List[str]:
        """Validate extracted output against Pan-TOP schema."""
        errors = []
        extractions = output.get('extractions', [])

        if not isinstance(extractions, list):
            errors.append("'extractions' must be a list")
            return errors

        valid_categories = set(EXTRACTION_CATEGORIES)
        for i, entry in enumerate(extractions):
            if not isinstance(entry, dict) or len(entry) != 1:
                errors.append(f'Entry {i}: must be a single-key dictionary')
                continue
            key = next(iter(entry))
            if key not in valid_categories:
                errors.append(f"Entry {i}: unknown category '{key}'")

        return errors
