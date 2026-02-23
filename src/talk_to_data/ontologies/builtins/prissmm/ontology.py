"""
PRISSMM Ontology Implementation

Implements OntologyBase for PRISSMM/GENIE BPC data model.
Provides structured extraction aligned with AACR Project GENIE Biopharma Collaborative.
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Any

from ...base import OntologyBase, DataItem, DataCategory

MODULE_DIR = Path(__file__).parent

SUPPORTED_COHORTS = {
    'nsclc': 'Non-Small Cell Lung Cancer',
    'lung': 'Non-Small Cell Lung Cancer',
    'breast': 'Breast Cancer',
    'colorectal': 'Colorectal Cancer',
    'crc': 'Colorectal Cancer',
    'generic': 'Other Cancer Type',
}

SITE_TO_COHORT = {
    'C34': 'nsclc',
    'C340': 'nsclc',
    'C341': 'nsclc',
    'C342': 'nsclc',
    'C343': 'nsclc',
    'C348': 'nsclc',
    'C349': 'nsclc',
    'C50': 'breast',
    'C500': 'breast',
    'C501': 'breast',
    'C502': 'breast',
    'C503': 'breast',
    'C504': 'breast',
    'C505': 'breast',
    'C506': 'breast',
    'C508': 'breast',
    'C509': 'breast',
    'C18': 'colorectal',
    'C19': 'colorectal',
    'C20': 'colorectal',
    'C180': 'colorectal',
    'C181': 'colorectal',
    'C182': 'colorectal',
    'C183': 'colorectal',
    'C184': 'colorectal',
    'C185': 'colorectal',
    'C186': 'colorectal',
    'C187': 'colorectal',
    'C188': 'colorectal',
    'C189': 'colorectal',
    'C199': 'colorectal',
    'C209': 'colorectal',
}


def _load_json_file(filepath):
    """Load a JSON file and return its contents."""
    if not filepath.exists():
        return {}
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def _normalize_site_code(primary_site: str) -> str:
    """Normalize primary site code."""
    if not primary_site:
        return ''
    return primary_site.upper().replace('.', '').replace(' ', '')


class PRISSMMOntology(OntologyBase):
    """
    PRISSMM/GENIE BPC Ontology Implementation.

    Provides data extraction aligned with AACR Project GENIE Biopharma Collaborative
    data dictionaries, including patient, cancer diagnosis, regimen, and
    medical oncologist assessment datasets with site-specific extensions.
    """

    def __init__(self):
        """Initialize and load data definitions."""
        self._base_data = self._load_base_data()
        self._site_specific = self._load_site_specific()

    def _load_base_data(self) -> dict:
        """Load base dataset definitions."""
        base_dir = MODULE_DIR / 'base'
        data = {}
        for filename in ('patient.json', 'cancer_diagnosis.json', 'regimen.json', 'medical_oncologist_assessment.json'):

            filepath = base_dir / filename
            if not filepath.exists():
                continue
            category = filename.replace('.json', '')
            data[category] = _load_json_file(filepath)
        return data

    def _load_site_specific(self) -> dict:
        """Load site-specific definitions."""
        site_dir = MODULE_DIR / 'site_specific'
        data = {}
        for filename in ('nsclc.json', 'breast.json', 'colorectal.json'):
            filepath = site_dir / filename
            if not filepath.exists():
                continue
            content = _load_json_file(filepath)
            schema_id = content.get('schema_id', filename.replace('.json', ''))
            data[schema_id] = content
        return data

    @property
    def ontology_id(self) -> str:
        return 'prissmm'

    @property
    def display_name(self) -> str:
        return 'PRISSMM/GENIE BPC'

    @property
    def version(self) -> str:
        return '2.0.0'

    @property
    def description(self) -> str:
        return 'GENIE BPC clinical data model with site-specific extensions'

    def get_base_items(self) -> List[DataCategory]:
        """Get base PRISSMM data categories."""
        categories = []

        if 'patient' in self._base_data:
            data = self._base_data['patient']

            items = [
                DataItem(
                    id=item.get('id', ''),
                    name=item.get('name', ''),
                    description=item.get('description', ''),
                    data_type=item.get('data_type', 'string'),
                    valid_values=item.get('valid_values'),
                    json_field=item.get('json_field'),
                )
                for item in data.get('items', [])
            ]

            categories.append(DataCategory(
                id='patient',
                name='Patient Dataset',
                description=data.get('description', ''),
                items=items,
                per_diagnosis=False,
            ))

        return categories

    def get_site_specific_items(self, cancer_type: str) -> List[DataCategory]:
        """Get cancer diagnosis, regimen, and site-specific items."""
        categories = []

        if 'cancer_diagnosis' in self._base_data:
            data = self._base_data['cancer_diagnosis']

            items = [
                DataItem(
                    id=item.get('id', ''),
                    name=item.get('name', ''),
                    description=item.get('description', ''),
                    data_type=item.get('data_type', 'string'),
                    valid_values=item.get('valid_values'),
                    json_field=item.get('json_field'),
                )
                for item in data.get('items', [])
            ]

            categories.append(DataCategory(
                id='cancer_diagnosis',
                name='Cancer Diagnosis Dataset',
                description=data.get('description', ''),
                items=items,
                per_diagnosis=True,
            ))

        cohort = cancer_type.lower()

        if cohort in self._site_specific:
            data = self._site_specific[cohort]

            items = [
                DataItem(
                    id=item.get('id', ''),
                    name=item.get('name', ''),
                    description=item.get('description', ''),
                    data_type=item.get('data_type', 'string'),
                    valid_values=item.get('valid_values'),
                    extraction_hints=item.get('extraction_hints', []),
                    json_field=item.get('json_field'),
                )
                for item in data.get('items', [])
            ]

            categories.append(DataCategory(
                id=f'site_specific_{cohort}',
                name=f'Site-Specific: {data.get("schema_name", cohort)}',
                description=data.get('description', ''),
                items=items,
                per_diagnosis=True,
            ))

        if 'regimen' in self._base_data:
            data = self._base_data['regimen']

            items = [
                DataItem(
                    id=item.get('id', ''),
                    name=item.get('name', ''),
                    description=item.get('description', ''),
                    data_type=item.get('data_type', 'string'),
                    valid_values=item.get('valid_values'),
                    json_field=item.get('json_field'),
                )
                for item in data.get('items', [])
            ]

            categories.append(DataCategory(
                id='regimen',
                name='Cancer-Directed Regimen Dataset',
                description=data.get('description', ''),
                items=items,
                per_diagnosis=True,
            ))

        if 'medical_oncologist_assessment' in self._base_data:
            data = self._base_data['medical_oncologist_assessment']

            items = [
                DataItem(
                    id=item.get('id', ''),
                    name=item.get('name', ''),
                    description=item.get('description', ''),
                    data_type=item.get('data_type', 'string'),
                    valid_values=item.get('valid_values'),
                    json_field=item.get('json_field'),
                )
                for item in data.get('items', [])
            ]

            categories.append(DataCategory(
                id='medical_oncologist_assessment',
                name='Medical Oncologist Assessment Dataset',
                description=data.get('description', ''),
                items=items,
                per_diagnosis=True,
            ))

        return categories

    def get_empty_summary_template(self) -> Dict[str, Any]:
        """Return empty PRISSMM JSON structure."""
        return {
            'version': self.version,
            'patient': {
                'birth_year': None,
                'naaccr_sex_code': None,
                'race_ethnicity': None,
                'age_at_diagnosis': None,
                'os_status': None,
                'os_months': None,
                'number_of_cancers': None,
            },
            'cancer_diagnoses': [],
            'regimens': [],
            'medical_oncologist_assessments': [],
        }

    def get_empty_diagnosis_template(self, cancer_type: str) -> Dict[str, Any]:
        """Return empty per-diagnosis structure."""
        template = {
            'ca_seq': None,
            'cohort': SUPPORTED_COHORTS.get(cancer_type.lower(), 'Other'),
            'ca_type': None,
            'naaccr_histology_code': None,
            'ca_hist_adeno_squamous': None,
            'naaccr_primary_site': None,
            'ca_stage': None,
            'ca_tnm_t': None,
            'ca_tnm_n': None,
            'ca_tnm_m': None,
            'ca_dmets_yn': None,
            'dmets_brain': None,
            'dmets_bone': None,
            'dmets_liver': None,
            'dmets_lung': None,
            'dmets_lymph': None,
            'dmets_adrenal': None,
            **{
                'dmets_pleura': None,
                'dmets_peritoneum': None,
                'dmets_other': None,
            },
        }

        cohort = cancer_type.lower()
        if cohort in ('nsclc', 'lung'):
            template.update({
                'ca_lung_cigarette': None,
                'ca_lung_pack_years': None,
                'egfr_status': None,
                'egfr_variant': None,
                'alk_status': None,
                'ros1_status': None,
                'kras_status': None,
                'braf_status': None,
                'pdl1_tps': None,
                'pdl1_category': None,
            })
            return template
        elif cohort == 'breast':
            template.update({
                'er_status': None,
                'pr_status': None,
                'her2_status': None,
                'receptor_subtype': None,
                'ki67': None,
                'grade': None,
                'brca1_status': None,
                'brca2_status': None,
                'oncotype_dx': None,
            })
            return template
        elif cohort in ('colorectal', 'crc'):
            template.update({
                'tumor_location': None,
                'tumor_sidedness': None,
                'kras_status': None,
                'nras_status': None,
                'braf_status': None,
                'msi_status': None,
                'her2_status': None,
                'ras_wild_type': None,
            })

        return template

    def format_for_prompt(self, cancer_type: str = "generic") -> str:
        """Format PRISSMM items for LLM prompts."""
        lines = []
        cohort_name = SUPPORTED_COHORTS.get(cancer_type.lower(), cancer_type)

        lines.append(f'Extract PRISSMM/GENIE BPC data for: {cohort_name}')
        lines.append('')

        # Patient dataset
        lines.append('=== PATIENT DATASET (one per patient) ===')
        lines.append('- birth_year: Year of birth')
        lines.append('- naaccr_sex_code: 1=Male, 2=Female')
        lines.append('- race_ethnicity: Combined race/ethnicity category')
        lines.append('- age_at_diagnosis: Age in years at first cancer diagnosis')
        lines.append('- os_status: 0=Alive, 1=Dead')
        lines.append('- os_months: Overall survival in months')
        lines.append('- number_of_cancers: Total number of cancer diagnoses')
        lines.append('')

        # Cancer diagnosis dataset
        lines.append('=== CANCER DIAGNOSIS DATASET (one per cancer, use ca_seq) ===')
        lines.append('- ca_seq: Cancer sequence number (0=first cancer)')
        lines.append('- cohort: Cancer cohort (NSCLC, Breast, CRC)')
        lines.append('- ca_type: Cancer type description')
        lines.append('- ca_hist_adeno_squamous: Histologic classification')
        lines.append('- ca_stage: AJCC stage (I, IA, IB, II, IIA, IIB, III, IIIA, IIIB, IIIC, IV, IVA, IVB)')
        lines.append('- ca_tnm_t/n/m: TNM components')
        lines.append('- ca_dmets_yn: Distant metastases present (Yes/No)')
        lines.append('- Metastatic sites (0=No, 1=Yes): dmets_brain, dmets_bone, dmets_liver,')
        lines.append('  dmets_lung, dmets_lymph, dmets_adrenal, dmets_pleura, dmets_peritoneum, dmets_other')
        lines.append('')

        # Site-specific sections
        cohort = cancer_type.lower()
        if cohort in ('nsclc', 'lung'):
            lines.append('=== NSCLC-SPECIFIC ===')
            lines.append('- ca_lung_cigarette: Smoking history (Current/Former/Never)')
            lines.append('- ca_lung_pack_years: Pack-years')
            lines.append('- ca_lung_pl_el_inv: Pleural invasion (PL 0/1/2/3)')
            lines.append('- Biomarkers:')
            lines.append('  - egfr_status: Mutated/Wild-type/Not tested')
            lines.append('  - egfr_variant: Exon 19 del, L858R, T790M, etc.')
            lines.append('  - alk_status, ros1_status, ret_status, met_status, ntrk_status')
            lines.append('  - kras_status, kras_variant (G12C, etc.), braf_status')
            lines.append('  - pdl1_tps: Percentage, pdl1_category: High/Low/Negative')
            lines.append('')
        elif cohort == 'breast':
            lines.append('=== BREAST-SPECIFIC ===')
            lines.append('- Receptor status:')
            lines.append('  - er_status: Positive/Negative, er_percent')
            lines.append('  - pr_status: Positive/Negative, pr_percent')
            lines.append('  - her2_status: Positive/Negative/Low, her2_ihc (0/1+/2+/3+)')
            lines.append('  - receptor_subtype: HR+/HER2-, HR+/HER2+, HR-/HER2+, Triple-negative')
            lines.append('- ki67: Proliferation index %')
            lines.append('- grade: 1/2/3 (Nottingham)')
            lines.append('- brca1_status, brca2_status: Mutation status')
            lines.append('- pik3ca_status, esr1_status: Mutation status')
            lines.append('- oncotype_dx: Recurrence score (0-100)')
            lines.append('- mammaprint: High risk/Low risk')
            lines.append('')
        elif cohort in ('colorectal', 'crc'):
            lines.append('=== CRC-SPECIFIC ===')
            lines.append('- tumor_location: Right colon/Left colon/Rectum')
            lines.append('- tumor_sidedness: Right-sided/Left-sided')
            lines.append('- Biomarkers:')
            lines.append('  - kras_status, kras_variant, nras_status')
            lines.append('  - braf_status: V600E/Other/Wild-type')
            lines.append('  - msi_status: MSI-H/MSS/MSI-L')
            lines.append('  - mmr_status: dMMR/pMMR')
            lines.append('  - her2_status')
            lines.append('  - ras_wild_type: Yes if both KRAS and NRAS wild-type')
            lines.append('  - ntrk_fusion')
            lines.append('- cea_baseline: CEA level at diagnosis')
            lines.append('')

        # Regimen dataset
        lines.append('=== REGIMEN DATASET (multiple per cancer, linked by ca_seq) ===')
        lines.append('- ca_seq: Cancer sequence this regimen treats')
        lines.append('- regimen_number: Sequential number (1=first regimen)')
        lines.append('- regimen_drugs: Drug names')
        lines.append('- drugs_num: Number of drugs')
        lines.append('- regimen_setting: Neoadjuvant/Adjuvant/First-line metastatic/Subsequent/Maintenance')
        lines.append('- dx_reg_start_days, dx_reg_end_days: Days from diagnosis')
        lines.append('- regimen_ongoing: Yes/No')
        lines.append('- regimen_disc_reason: Completed/Progression/Toxicity/Patient preference')
        lines.append('- includes_immunotherapy, includes_targeted, includes_chemo: 0/1 flags')
        lines.append('')

        # Medical oncologist assessment
        lines.append('=== MEDICAL ONCOLOGIST ASSESSMENT (longitudinal) ===')
        lines.append('- ca_seq: Cancer being assessed')
        lines.append('- md_dx_days: Days from diagnosis')
        lines.append('- md_ca: Evidence of cancer / No evidence of cancer')
        lines.append('- md_ca_status: Improving/Stable/Mixed/Worsening')
        lines.append('- md_ecog: ECOG performance status (0-4)')
        lines.append('')

        return '\n'.join(lines)

    def get_supported_cancer_types(self) -> List[str]:
        """List supported cohorts."""
        return list(SUPPORTED_COHORTS.keys())

    def detect_cancer_type(self, primary_site: str = None, histology: str = None, diagnosis_year: int = None) -> str:
        """Map site to PRISSMM cohort."""
        if not primary_site:
            return 'generic'

        site_code = _normalize_site_code(primary_site)

        if site_code in SITE_TO_COHORT:
            return SITE_TO_COHORT[site_code]

        for prefix_len in (3, 2):
            if len(site_code) >= prefix_len:
                prefix = site_code[:prefix_len]
                if prefix in SITE_TO_COHORT:
                    return SITE_TO_COHORT[prefix]

        return 'generic'

    def get_extraction_context(self) -> str:
        """Additional context for PRISSMM extraction."""
        return (
            "PRISSMM/GENIE BPC is a structured data model used by AACR Project GENIE\n"
            "for biopharma collaborative research. Key characteristics:\n"
            "1. Patient-level data (demographics, survival) is separate from per-diagnosis data\n"
            "2. Cancer diagnoses are tracked with ca_seq (0=first, 1=second, etc.)\n"
            "3. Regimens are linked to specific cancers via ca_seq\n"
            "4. Medical oncologist assessments track disease status over time\n"
            "5. Site-specific biomarkers vary by cohort (NSCLC, Breast, CRC)\n"
            "\n"
            "For metastatic sites, use 0=No, 1=Yes for dmets_* fields."
        )
