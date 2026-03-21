"""
Generic Cancer Ontology Implementation

Implements OntologyBase for cancer-type-agnostic data extraction from clinical notes.
Provides structured extraction across all cancer types using broad field definitions.

Categories:
- cancer_diagnosis
- cancer_biomarker
- cancer_systemic_therapy_regimen
- cancer_surgery
- cancer_radiation_therapy
- cancer_burden
- social_history
"""

import json
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
    'social_history',
]


def _load_json_file(filepath: Path) -> Dict:
    if not filepath.exists():
        return {}
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


class GenericCancerOntology(OntologyBase):
    """
    Generic Cancer Ontology Implementation.

    Extracts structured data from clinical notes for cancer patients of any
    type, covering diagnosis, biomarkers, systemic therapy, surgery,
    radiation, disease burden, and social history.
    """

    @property
    def ontology_id(self) -> str:
        return 'generic_cancer'

    @property
    def display_name(self) -> str:
        return 'Generic Cancer'

    @property
    def version(self) -> str:
        return '1.0.0'

    @property
    def description(self) -> str:
        return 'Cancer-type-agnostic extraction fields (diagnosis, biomarker, treatment, surgery, radiation, burden, social history)'

    def get_base_items(self) -> List[DataCategory]:
        """Load base data items from JSON files."""
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
                    repeatable=category_name not in ('social_history',),
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
        """Return per-diagnosis categories.

        This ontology is cancer-type agnostic so all categories apply
        regardless of cancer type.
        """
        return self.get_base_items()

    def get_empty_summary_template(self) -> Dict[str, Any]:
        """Return empty JSON structure."""
        return {'extractions': []}

    def get_empty_diagnosis_template(self, cancer_type: str = 'generic') -> Dict[str, Any]:
        """Return empty diagnosis entry structure."""
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
            'social_history': {
                'categorical_smoking_history': None,
                'pack_years': None,
                'alcohol_use': None,
                'substance_use': None,
            },
        }

    def format_for_prompt(self, cancer_type: str = "generic") -> str:
        """Format extraction instructions for LLM prompts."""
        lines = []
        lines.append('Extract structured oncology data from clinical notes.')
        lines.append(
            'Extract key information from clinical notes, imaging reports, and pathology reports for cancer patients.')

        lines.append('')
        lines.append(
            'The output should be a list of dictionaries. Each dictionary has a single key (the category name) whose value is a dictionary of subfields.')

        lines.append('')

        lines.append('=== CATEGORIES AND SUBFIELDS ===')
        lines.append('')

        lines.append('1. cancer_diagnosis')
        lines.append('   Subfields:')
        lines.append('   - primary_site (eg lung, breast, colon, prostate, pancreas, ovary, kidney, bladder, brain, liver, skin, thyroid)')
        lines.append('   - overall_stage_at_diagnosis (ie I, IIA, IIIB, IV)')
        lines.append('   - t_stage_at_diagnosis (ie T2a, T3b)')
        lines.append('   - n_stage_at_diagnosis (ie N0, N1, N2, N3)')
        lines.append('   - m_stage_at_diagnosis (ie M0, M1a, M1b)')
        lines.append(
            '   - histology (eg adenocarcinoma, squamous_cell_carcinoma, ductal_carcinoma, lobular_carcinoma, '
            'transitional_cell_carcinoma, renal_cell_carcinoma, hepatocellular_carcinoma, melanoma, glioblastoma, '
            'lymphoma, sarcoma, small_cell, large_cell, neuroendocrine, mesothelioma, urothelial, clear_cell, '
            'papillary, mucinous)')
        lines.append(
            '   - diagnosis_date (as specific as is known; if only year is indicated, list year; if month is indicated also, list month and year)')
        lines.append(
            '   Guidelines: One entry per cancer diagnosis; recurrences or metastases do not count as a new diagnosis. Be very careful not to include non-cancer diagnoses.')

        lines.append('')

        lines.append('2. cancer_biomarker')
        lines.append('   Subfields:')
        lines.append('   - biomarker_type (eg mutation, rearrangement, expression, amplification, serum_marker)')
        lines.append(
            '   - biomarker_tested (eg ER, PR, HER2, BRCA1, BRCA2, EGFR, ALK, ROS1, PD-L1, KRAS, NRAS, BRAF, '
            'PIK3CA, MSI, MMR, TMB, NTRK, RET, MET, FGFR, IDH1, IDH2, PSA, CEA, CA-125, AFP, Ki-67, '
            'TTF-1, p40, BCR-ABL). Note: EGFR refers to EGFR mutation only, not estimated GFR.')
        lines.append(
            '   - biomarker_result (eg positive, negative, 60%, L858R mutant, MSI-high, ER positive 95%)')

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
            '   - surgery_type (eg mastectomy, lumpectomy, colectomy, prostatectomy, nephrectomy, cystectomy, '
            'hysterectomy, craniotomy, lobectomy, wedge_resection, gastrectomy, pancreatectomy, hepatectomy, '
            'thyroidectomy, lymph_node_dissection, wide_local_excision, orchiectomy, oophorectomy, debulking)')
        lines.append(
            '   - surgery_site (eg breast, colon, prostate, kidney, lung, brain, liver, pancreas, ovary, skin, lymph_node, axilla)')
        lines.append('   - surgery_date (as specific as is known)')
        lines.append('')

        lines.append('5. cancer_radiation_therapy')
        lines.append('   Subfields:')
        lines.append('   - radiation_type (eg external_beam, SBRT, SRS, brachytherapy, proton, IMRT, whole_brain)')
        lines.append('   - radiation_start_date (as specific as is known)')
        lines.append('   - radiation_end_date (as specific as is known)')
        lines.append('   - radiation_site (eg brain, bone, liver, lung, pelvis, breast, prostate, chest, head_and_neck, spine)')
        lines.append('')

        lines.append('6. cancer_burden')
        lines.append('   Subfields:')
        lines.append('   - burden_assessor_type (eg radiologist, oncologist)')
        lines.append('   - burden_assessment_date')
        lines.append('   - burden_type (eg response/improvement, progression/worsening, cancer_site). If cancer_site, add another cancer_site subfield to specify the location.')
        lines.append('   Guidelines: Progression/worsening should only be called if the tumor is growing.')
        lines.append('')

        lines.append('7. social_history')
        lines.append('   Subfields:')
        lines.append('   - categorical_smoking_history (eg never, former, current)')
        lines.append('   - pack_years (extract whatever was documented)')
        lines.append('   - alcohol_use (eg never, former, current, social)')
        lines.append('   - substance_use (extract whatever was documented)')
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
        lines.append('  {"cancer_diagnosis": {"primary_site": "breast", "histology": "ductal_carcinoma", "overall_stage_at_diagnosis": "IIA"}},')
        lines.append('  {"cancer_biomarker": {"biomarker_type": "expression", "biomarker_tested": "ER", "biomarker_result": "positive 95%"}},')
        lines.append('  {"cancer_biomarker": {"biomarker_type": "expression", "biomarker_tested": "HER2", "biomarker_result": "negative"}},')
        lines.append('  {"cancer_systemic_therapy_regimen": {"regimen_drug_list": ["doxorubicin", "cyclophosphamide"]}},')
        lines.append('  {"social_history": {"categorical_smoking_history": "never", "alcohol_use": "social"}}')
        lines.append(']')

        return '\n'.join(lines)

    def get_supported_cancer_types(self) -> List[str]:
        """List supported cancer types."""
        return ['generic']

    def detect_cancer_type(self, primary_site: Optional[str] = None,
                           histology: Optional[str] = None,
                           diagnosis_year: Optional[int] = None) -> str:
        """Always returns generic — this ontology is cancer-type agnostic."""
        return 'generic'

    def get_extraction_context(self) -> str:
        """Additional context for extraction."""
        return (
            'Generic Cancer ontology extracts structured data from clinical notes '
            'for cancer patients of any type. The extraction schema covers diagnosis, '
            'biomarkers, systemic therapy, surgery, radiation, disease burden, and '
            'social history. Each extraction entry is a single-key dictionary with '
            'the category name as key and subfields as value. Multiple entries of the '
            'same category are allowed (e.g., multiple biomarkers, multiple regimens).'
        )

    def validate_output(self, output: Dict[str, Any]) -> List[str]:
        """Validate extracted output against schema."""
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
