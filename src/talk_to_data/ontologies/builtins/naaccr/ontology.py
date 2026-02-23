"""
NAACCR Ontology Implementation

Implements OntologyBase for NAACCR v25 cancer registry standards.
Includes site-specific data items (SSDIs) for multiple cancer types.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any

from ...base import OntologyBase, DataItem, DataCategory


MODULE_DIR = Path(__file__).parent


SUPPORTED_SCHEMAS = [
    'lung', 'lung_v9', 'breast', 'prostate', 'colorectal',
    'melanoma', 'kidney', 'bladder', 'pancreas', 'head_neck',
    'thyroid', 'liver', 'esophagus', 'stomach', 'ovary',
    'uterus', 'cervix', 'testis', 'brain', 'lymphoma',
    'leukemia', 'myeloma', 'generic',
]

SCHEMA_DISPLAY_NAMES = {
    'lung': 'Lung Cancer (2018-2024)',
    'lung_v9': 'Lung Cancer (2025+)',
    'breast': 'Breast Cancer',
    'prostate': 'Prostate Cancer',
    'colorectal': 'Colorectal Cancer',
    'melanoma': 'Melanoma (Cutaneous)',
    'kidney': 'Kidney/Renal Cell Cancer',
    'bladder': 'Bladder Cancer',
    'pancreas': 'Pancreatic Cancer',
    'head_neck': 'Head and Neck Cancer',
    'thyroid': 'Thyroid Cancer',
    'liver': 'Hepatocellular Carcinoma',
    'esophagus': 'Esophageal Cancer',
    'stomach': 'Gastric Cancer',
    'ovary': 'Ovarian Cancer',
    'uterus': 'Uterine/Endometrial Cancer',
    'cervix': 'Cervical Cancer',
    'testis': 'Testicular Cancer',
    'brain': 'Brain/CNS Tumors',
    'lymphoma': 'Lymphoma',
    'leukemia': 'Leukemia',
    'myeloma': 'Multiple Myeloma',
    'generic': 'Other/Unspecified Cancer',
}

SITE_TO_SCHEMA_MAP = {
    # Lung C34
    'C340': 'lung', 'C341': 'lung', 'C342': 'lung',
    'C343': 'lung', 'C348': 'lung', 'C349': 'lung', 'C34': 'lung',
    # Breast C50
    'C500': 'breast', 'C501': 'breast', 'C502': 'breast',
    'C503': 'breast', 'C504': 'breast', 'C505': 'breast',
    'C506': 'breast', 'C508': 'breast', 'C509': 'breast', 'C50': 'breast',
    # Prostate C61
    'C619': 'prostate', 'C61': 'prostate',
    # Colorectal C18-C20
    'C180': 'colorectal', 'C181': 'colorectal', 'C182': 'colorectal',
    'C183': 'colorectal', 'C184': 'colorectal', 'C185': 'colorectal',
    'C186': 'colorectal', 'C187': 'colorectal', 'C188': 'colorectal',
    'C189': 'colorectal', 'C18': 'colorectal',
    'C199': 'colorectal', 'C19': 'colorectal',
    'C209': 'colorectal', 'C20': 'colorectal',
    # Kidney C64
    'C649': 'kidney', 'C64': 'kidney',
    # Bladder C67
    'C670': 'bladder', 'C671': 'bladder', 'C672': 'bladder',
    'C673': 'bladder', 'C674': 'bladder', 'C675': 'bladder',
    'C676': 'bladder', 'C677': 'bladder', 'C678': 'bladder',
    'C679': 'bladder', 'C67': 'bladder',
    # Pancreas C25
    'C250': 'pancreas', 'C251': 'pancreas', 'C252': 'pancreas',
    'C253': 'pancreas', 'C254': 'pancreas', 'C257': 'pancreas',
    'C258': 'pancreas', 'C259': 'pancreas', 'C25': 'pancreas',
    # Thyroid C73
    'C739': 'thyroid', 'C73': 'thyroid',
    # Liver C22
    'C220': 'liver', 'C221': 'liver', 'C22': 'liver',
    # Esophagus C15
    'C150': 'esophagus', 'C151': 'esophagus', 'C152': 'esophagus',
    'C153': 'esophagus', 'C154': 'esophagus', 'C155': 'esophagus',
    'C158': 'esophagus', 'C159': 'esophagus', 'C15': 'esophagus',
    # Stomach C16
    'C160': 'stomach', 'C161': 'stomach', 'C162': 'stomach',
    'C163': 'stomach', 'C164': 'stomach', 'C165': 'stomach',
    'C166': 'stomach', 'C168': 'stomach', 'C169': 'stomach', 'C16': 'stomach',
    # Ovary C56
    'C569': 'ovary', 'C56': 'ovary',
    # Uterus C54-C55
    'C540': 'uterus', 'C541': 'uterus', 'C542': 'uterus',
    'C543': 'uterus', 'C548': 'uterus', 'C549': 'uterus',
    'C54': 'uterus', 'C559': 'uterus', 'C55': 'uterus',
    # Cervix C53
    'C530': 'cervix', 'C531': 'cervix', 'C538': 'cervix',
    'C539': 'cervix', 'C53': 'cervix',
    # Testis C62
    'C620': 'testis', 'C621': 'testis', 'C629': 'testis', 'C62': 'testis',
    # Brain C71
    'C710': 'brain', 'C711': 'brain', 'C712': 'brain',
    'C713': 'brain', 'C714': 'brain', 'C715': 'brain',
    'C716': 'brain', 'C717': 'brain', 'C718': 'brain',
    'C719': 'brain', 'C71': 'brain',
    # Head and Neck C00-C14, C30-C32
    'C00': 'head_neck', 'C01': 'head_neck', 'C02': 'head_neck',
    'C03': 'head_neck', 'C04': 'head_neck', 'C05': 'head_neck',
    'C06': 'head_neck', 'C07': 'head_neck',
    'C08': 'head_neck', 'C09': 'head_neck', 'C10': 'head_neck',
    'C11': 'head_neck', 'C12': 'head_neck', 'C13': 'head_neck',
    'C14': 'head_neck', 'C30': 'head_neck', 'C31': 'head_neck',
    'C32': 'head_neck',
}

MELANOMA_HISTOLOGIES = frozenset({
    '8720', '8721', '8722', '8723', '8725', '8726', '8727', '8728',
    '8730', '8740', '8741', '8742', '8743', '8744', '8745', '8746',
    '8761', '8770', '8771', '8772', '8773', '8774', '8780',
})

LYMPHOMA_HISTOLOGIES = frozenset({
    '9590', '9591', '9596', '9597', '9650', '9651', '9652', '9653',
    '9654', '9655', '9659', '9661', '9662', '9663', '9664', '9665',
    '9667', '9670', '9671', '9673', '9675', '9678', '9679', '9680',
    '9684', '9687', '9688', '9689', '9690', '9691', '9695', '9698',
    '9699', '9700', '9701', '9702', '9705', '9708', '9709', '9712',
    '9714', '9716', '9717', '9718', '9719', '9724', '9725', '9726',
    '9727', '9728', '9729',
})

LEUKEMIA_HISTOLOGIES = frozenset({
    '9800', '9801', '9805', '9806', '9807', '9808', '9809', '9811',
    '9812', '9813', '9814', '9815', '9816', '9817', '9818', '9820',
    '9823', '9826', '9827', '9831', '9832', '9833', '9834', '9835',
    '9836', '9837', '9840', '9860', '9861', '9863', '9865', '9866',
    '9867', '9869', '9870', '9871', '9872', '9873', '9874', '9875',
    '9876', '9891', '9895', '9896', '9897', '9898', '9910', '9911',
    '9920', '9930', '9931', '9940', '9945', '9946', '9948', '9963',
    '9964', '9965', '9966', '9967', '9975',
})

MYELOMA_HISTOLOGIES = frozenset({
    '9731', '9732', '9733', '9734',
})


def _normalize_site_code(primary_site: str) -> str:
    """Normalize ICD-O-3 site code."""
    if not primary_site:
        return ''
    return primary_site.upper().replace('.', '').replace(' ', '')


def _load_json_file(filepath: Path) -> Dict:
    """Load JSON file, returns empty dict if not exists."""
    if not filepath.exists():
        return {}
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


class NAACCROntology(OntologyBase):
    """
    NAACCR v25 Ontology Implementation.

    Provides cancer registry data items following NAACCR Version 25 standards.
    Supports site-specific data items (SSDIs) for multiple cancer types.
    """

    @property
    def ontology_id(self) -> str:
        return 'naaccr'

    @property
    def display_name(self) -> str:
        return 'NAACCR v25 Cancer Registry'

    @property
    def version(self) -> str:
        return '25.0'

    @property
    def description(self) -> str:
        return 'North American cancer registry fields with site-specific items'

    def get_base_items(self) -> List[DataCategory]:
        """Load base NAACCR data items from JSON files."""
        categories = []

        base_dir = MODULE_DIR / 'base'
        for filename in ('demographics.json', 'tumor_identification.json', 'staging.json', 'treatment.json'):

            filepath = base_dir / filename
            if not filepath.exists():
                continue
            data = _load_json_file(filepath)
            category_id = filename.replace('.json', '')
            category_name = data.get('category', category_id).replace('_', ' ').title()

            items = []
            for item_data in data.get('items', []):
                items.append(DataItem(
                    id=item_data.get('json_field', f"naaccr_{item_data.get('naaccr_item', '')}"),
                    name=item_data.get('name', ''),
                    description=item_data.get('description', ''),
                    data_type=item_data.get('data_type', 'string'),
                    valid_values=None,
                    extraction_hints=item_data.get('extraction_hints', []),
                    json_field=item_data.get('json_field'),
                ))

            categories.append(DataCategory(
                id=category_id,
                name=category_name,
                description=data.get('description', ''),
                items=items,
                per_diagnosis=(category_id != 'demographics'),
            ))

        return categories

    def get_site_specific_items(self, cancer_type: str) -> List[DataCategory]:
        """Load site-specific data items for a cancer type."""
        site_dir = MODULE_DIR / 'site_specific'

        filepath = site_dir / f'{cancer_type}.json'
        if not filepath.exists():
            filepath = site_dir / 'generic.json'
            if not filepath.exists():
                return []

        data = _load_json_file(filepath)
        if not data:
            return []

        items = []
        for item_data in data.get('items', []):
            items.append(DataItem(
                id=item_data.get('json_field', f"naaccr_{item_data.get('naaccr_item', '')}"),
                name=item_data.get('name', ''),
                description=item_data.get('description', ''),
                data_type='string',
                valid_values=None,
                extraction_hints=item_data.get('extraction_hints', []),
                json_field=item_data.get('json_field'),
            ))

        schema_name = data.get('schema_name', SCHEMA_DISPLAY_NAMES.get(cancer_type, cancer_type))

        return [DataCategory(
            id=f'site_specific_{cancer_type}',
            name=f'Site-Specific Data Items ({schema_name})',
            description=data.get('description', ''),
            items=items,
            per_diagnosis=True,
        )]

    def get_empty_summary_template(self) -> Dict[str, Any]:
        """Return empty patient summary template."""
        patient_fields = (
            'naaccr_220_sex', 'sex_at_birth',
            'naaccr_240_date_of_birth', 'date_of_birth',
            'naaccr_160_race1', 'race',
            'naaccr_190_spanish_hispanic_origin', 'ethnicity',
        )
        return {
            'patient': {k: None for k in patient_fields},
            'diagnoses': [],
        }

    def get_empty_diagnosis_template(self, cancer_type: str = 'generic') -> Dict[str, Any]:
        """Return empty diagnosis template."""
        schema_name = SCHEMA_DISPLAY_NAMES.get(cancer_type, cancer_type)

        return {
            'diagnosis_index': None,
            'naaccr_version': '25',
            'schema_id': cancer_type,
            'schema_name': schema_name,
            # Primary site / tumor identification
            'naaccr_400_primary_site': None,
            'primary_site': None,
            'naaccr_410_laterality': None,
            'laterality': None,
            'naaccr_420_histologic_type': None,
            'histology': None,
            'naaccr_430_behavior': None,
            'behavior': None,
            'naaccr_440_grade': None,
            'grade': None,
            'naaccr_390_date_of_diagnosis': None,
            'date_of_diagnosis': None,
            'naaccr_250_age_at_diagnosis': None,
            'age_at_diagnosis': None,
            # Staging
            'staging': {
                'naaccr_760_tnm_edition': None,
                'ajcc_edition': None,
                'naaccr_940_tnm_clin_t': None,
                'clinical_t': None,
                'naaccr_950_tnm_clin_n': None,
                'clinical_n': None,
                'naaccr_960_tnm_clin_m': None,
                'clinical_m': None,
                'naaccr_970_tnm_clin_stage_group': None,
                'clinical_stage_group': None,
                'naaccr_880_tnm_path_t': None,
                'pathologic_t': None,
                'naaccr_890_tnm_path_n': None,
                'pathologic_n': None,
                'naaccr_900_tnm_path_m': None,
                'pathologic_m': None,
                'naaccr_910_tnm_path_stage_group': None,
                'pathologic_stage_group': None,
                'naaccr_759_seer_summary_stage_2018': None,
                'seer_summary_stage': None,
            },
            # Site-specific factors placeholder
            'site_specific_factors': {},
            # First course treatment
            'first_course_treatment': {
                'naaccr_1290_surgical_procedure': None,
                'surgical_procedure': None,
                'naaccr_1310_surgical_margins': None,
                'surgical_margins': None,
                'naaccr_1350_regional_nodes_examined': None,
                'nodes_examined': None,
                'naaccr_1360_regional_nodes_positive': None,
                'nodes_positive': None,
                'naaccr_1420_radiation': None,
                'radiation': None,
                'naaccr_1520_chemotherapy': None,
                'chemotherapy': None,
                'naaccr_1560_immunotherapy': None,
                'immunotherapy': None,
            },
        }

    def format_for_prompt(self, cancer_type: str) -> str:
        """Format extraction instructions for LLM prompts."""
        lines = []
        schema_name = SCHEMA_DISPLAY_NAMES.get(cancer_type, cancer_type)

        lines.append(f'Extract NAACCR v25 data items for: {schema_name}')
        lines.append('Include both NAACCR item codes (naaccr_XXX_fieldname) and human-readable values.')
        lines.append('')

        # Demographics
        lines.append('=== DEMOGRAPHICS (Patient-level) ===')
        lines.append('- [220] Sex: 1=Male, 2=Female, 3=Other, 9=Unknown')
        lines.append('- [240] Date of Birth: YYYYMMDD format')
        lines.append('- [160] Race: 01=White, 02=Black, 03=AI/AN, etc.')
        lines.append('- [190] Hispanic Origin: 0=Non-Hispanic, 1-8=Hispanic subtypes, 9=Unknown')
        lines.append('')

        # Tumor identification
        lines.append('=== TUMOR IDENTIFICATION (Per-diagnosis) ===')
        lines.append('- [400] Primary Site: ICD-O-3 topography (C##.#)')
        lines.append('- [410] Laterality: 0=Not paired, 1=Right, 2=Left, 3=One side unknown, 4=Bilateral, 9=Unknown')
        lines.append('- [420] Histologic Type: ICD-O-3 morphology (4 digits)')
        lines.append('- [430] Behavior: 0=Benign, 1=Uncertain, 2=In situ, 3=Malignant primary')
        lines.append('- [440] Grade: 1=Well diff, 2=Mod diff, 3=Poorly diff, 4=Undiff, 9=Unknown')
        lines.append('- [390] Date of Diagnosis: YYYYMMDD format')
        lines.append('- [250] Age at Diagnosis: Years (000-120, 999=Unknown)')
        lines.append('')

        # Staging
        lines.append('=== STAGING (Per-diagnosis) ===')
        lines.append('- [760] TNM Edition: 07=7th, 08=8th')
        lines.append('- [940-960] Clinical T/N/M')
        lines.append('- [970] Clinical Stage Group')
        lines.append('- [880-900] Pathologic T/N/M')
        lines.append('- [910] Pathologic Stage Group')
        lines.append('- [759] SEER Summary Stage 2018: 0=In situ, 1=Localized, 7=Distant')
        lines.append('')

        # Treatment
        lines.append('=== FIRST COURSE TREATMENT (Per-diagnosis) ===')
        lines.append('- [1290] Surgical Procedure: 00=None, 10-80=Specific procedures')
        lines.append('- [1310] Surgical Margins: 0=R0, 2=R1, 3=R2')
        lines.append('- [1350] Regional Nodes Examined: Number')
        lines.append('- [1360] Regional Nodes Positive: Number')
        lines.append('- [1420] Radiation: 0=None, 1=Beam, 2=Implants')
        lines.append('- [1520] Chemotherapy: 00=None, 01=Single agent, 02=Multiple agents')
        lines.append('- [1560] Immunotherapy: 00=None, 01=Given')
        lines.append('')

        # Site-specific items
        site_items = self.get_site_specific_items(cancer_type)
        if site_items:
            for category in site_items:
                lines.append(f'=== {category.name.upper()} ===')
                for item in category.items:
                    lines.append(f'- {item.name}: {item.description}')
                    if item.extraction_hints:
                        lines.append(f'  Look for: {", ".join(item.extraction_hints[:5])}')
                lines.append('')

        return '\n'.join(lines)

    def get_supported_cancer_types(self) -> List[str]:
        """Return list of supported cancer types."""
        return SUPPORTED_SCHEMAS

    def detect_cancer_type(self, primary_site: Optional[str] = None,
                           histology: Optional[str] = None,
                           diagnosis_year: Optional[int] = None) -> str:
        """Detect cancer type from primary site and histology codes."""
        # Normalize site code
        site_code = _normalize_site_code(primary_site or '')

        # Get first 4 digits of histology
        histology_code = (histology or '')[:4]

        # Check histology-based types first
        if histology_code in MYELOMA_HISTOLOGIES:
            return 'myeloma'
        if histology_code in LEUKEMIA_HISTOLOGIES:
            return 'leukemia'
        if histology_code in LYMPHOMA_HISTOLOGIES:
            return 'lymphoma'

        # Melanoma: skin site + melanoma histology
        if site_code.startswith('C44') and histology_code in MELANOMA_HISTOLOGIES:
            return 'melanoma'

        # Direct site code lookup
        if site_code in SITE_TO_SCHEMA_MAP:
            schema = SITE_TO_SCHEMA_MAP[site_code]
            if schema == 'lung' and diagnosis_year and diagnosis_year >= 2025:
                return 'lung_v9'
            return schema

        # Try prefix matching (3 chars, then 2 chars)
        for prefix_len in (3, 2):
            if len(site_code) >= prefix_len:
                prefix = site_code[:prefix_len]
                if prefix in SITE_TO_SCHEMA_MAP:
                    schema = SITE_TO_SCHEMA_MAP[prefix]
                    if schema == 'lung' and diagnosis_year and diagnosis_year >= 2025:
                        return 'lung_v9'
                    return schema

        return 'generic'

    def get_extraction_context(self) -> str:
        """Return NAACCR-specific context for prompts."""
        return (
            'NAACCR (North American Association of Central Cancer Registries) provides\n'
            'standardized data items for cancer registration. Use NAACCR item numbers (e.g., [400])\n'
            'for coding. Include both coded values and human-readable descriptions.\n'
            'Per-diagnosis fields should be tracked separately for each cancer diagnosis.'
        )
