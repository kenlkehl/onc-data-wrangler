"""
MSK-CHORD Ontology Implementation

Implements OntologyBase for MSK-CHORD (Clinical Health Oncology Research Data),
the clinical data model used for the MSK-IMPACT cBioPortal study.

MSK-CHORD provides:
- Patient summary data (demographics, tumor sites, biomarkers)
- Sample summary data (genomic, cancer type, MSI/TMB)
- Timeline data (longitudinal clinical events)

Reference: https://cdsi.mskcc.org/
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Any

from ...base import OntologyBase, DataItem, DataCategory

MODULE_DIR = Path(__file__).parent


def _load_json_file(filepath):
    """Load a JSON file and return its contents."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


ONCOTREE_MAPPINGS = {
    'lung': ('LUAD', 'LUSC', 'NSCLC', 'SCLC', 'LUNE', 'LUNG'),
    'breast': ('BRCA', 'IDC', 'ILC', 'BREAST'),
    'colorectal': ('COAD', 'READ', 'CRC', 'COADREAD'),
    'prostate': ('PRAD', 'PROSTATE'),
    'melanoma': ('MEL', 'SKCM', 'UM'),
    'pancreas': ('PAAD', 'PANET', 'PANCREAS'),
    'kidney': ('CCRCC', 'PRCC', 'CHRCC', 'RCC', 'KIDNEY'),
    'bladder': ('BLCA', 'BLADDER'),
    'ovarian': ('OV', 'HGSOC', 'OVARY'),
    'endometrial': ('UCEC', 'UCS', 'UTERUS'),
    'glioma': ('GBM', 'LGG', 'BRAIN'),
    'head_neck': ('HNSC', 'HEADNECK'),
    'thyroid': ('THCA', 'THYROID'),
    'liver': ('HCC', 'LIHC', 'LIVER'),
    'gastric': ('STAD', 'STOMACH'),
    'esophageal': ('ESCA', 'ESOPHAGUS'),
    'sarcoma': ('SARC', 'SARCOMA'),
    'leukemia': ('AML', 'ALL', 'CLL', 'CML', 'LEUKEMIA'),
    'lymphoma': ('DLBCL', 'FL', 'HL', 'NHL', 'LYMPHOMA'),
    'myeloma': ('MM', 'MYELOMA'),
}


class MSKChordOntology(OntologyBase):
    """
    MSK-CHORD Clinical Data Model Ontology.

    Provides data extraction aligned with the MSK-IMPACT cBioPortal clinical
    data model, including patient summary, sample summary, and timeline data.
    """

    def __init__(self):
        """Initialize and load data definitions."""
        self._patient_summary = _load_json_file(MODULE_DIR / 'base' / 'patient_summary.json')
        self._sample_summary = _load_json_file(MODULE_DIR / 'base' / 'sample_summary.json')
        self._timeline_data = self._load_timeline_data()

    def _load_timeline_data(self):
        """Load all timeline data definitions."""
        timeline_dir = MODULE_DIR / 'timeline'
        data = {}
        for filepath in timeline_dir.glob('*.json'):
            category_data = _load_json_file(filepath)
            data[category_data.get('category_id', filepath.stem)] = category_data
        return data

    @property
    def ontology_id(self) -> str:
        return 'msk_chord'

    @property
    def display_name(self) -> str:
        return 'MSK-CHORD (cBioPortal Clinical Data Model)'

    @property
    def version(self) -> str:
        return '2023.12'

    def get_base_items(self) -> List[DataCategory]:
        """Get patient-level MSK-CHORD items."""
        categories = []

        patient_items = []
        for item in self._patient_summary.get('items', []):
            patient_items.append(DataItem(
                id=item['id'],
                name=item['name'],
                description=item.get('description', ''),
                data_type=item.get('data_type', 'string'),
                valid_values=item.get('valid_values'),
                extraction_hints=[item.get('column_name')],
            ))

        categories.append(DataCategory(
            id='patient_summary',
            name='Patient Summary',
            description='Patient-level clinical attributes',
            items=patient_items,
            per_diagnosis=False,
        ))

        tumor_site_items = []
        for item in self._patient_summary.get('tumor_site_items', []):
            tumor_site_items.append(DataItem(
                id=item['id'],
                name=item['name'],
                description=item.get('description', ''),
                data_type='boolean',
                valid_values=item.get('valid_values'),
            ))

        categories.append(DataCategory(
            id='tumor_site_summary',
            name='Tumor Site Summary (NLP)',
            description='Patient-level tumor site history from radiology',
            items=tumor_site_items,
            per_diagnosis=False,
        ))

        return categories

    def get_site_specific_items(self, cancer_type: str) -> List[DataCategory]:
        """Get sample-level and timeline items."""
        categories = []

        if self._sample_summary:
            sample_items = []
            for item in self._sample_summary.get('items', []):
                sample_items.append(DataItem(
                    id=item['id'],
                    name=item['name'],
                    description=item.get('description', ''),
                    data_type=item.get('data_type', 'string'),
                    valid_values=item.get('valid_values'),
                ))

            categories.append(DataCategory(
                id='sample_summary',
                name='Sample Summary',
                description='Sample-level clinical and genomic attributes',
                items=sample_items,
                per_diagnosis=True,
            ))

        for cat_id, cat_data in self._timeline_data.items():
            timeline_items = []
            for item in cat_data.get('items', []):
                timeline_items.append(DataItem(
                    id=item['id'],
                    name=item['name'],
                    description=item.get('description', ''),
                    data_type=item.get('data_type', 'string'),
                    valid_values=item.get('valid_values'),
                    repeatable=True,
                ))

            if timeline_items:
                categories.append(DataCategory(
                    id=cat_id,
                    name=cat_data.get('category_name', cat_id),
                    description=cat_data.get('description', ''),
                    items=timeline_items,
                    per_diagnosis=True,
                ))

        return categories

    def get_empty_summary_template(self) -> Dict[str, Any]:
        """Return empty MSK-CHORD JSON structure."""
        return {
            'version': self.version,
            'patient': {
                'patient_id': None,
                'gender': None,
                'race': None,
                'ethnicity': None,
                'current_age': None,
                'ancestry_label': None,
                'smoking_history': None,
                'ecog_kps_first': None,
                'bmi_first': None,
                'prior_treatment_to_msk': None,
                'history_of_pdl1_positive': None,
                'history_of_dmmr': None,
                'gleason_first_reported': None,
                'gleason_highest_reported': None,
                'stage_highest_recorded': None,
                'os_months': None,
                'os_status': None,
            },
            'tumor_sites_summary': {
                'adrenal_glands': None,
                'bone': None,
                'cns_brain': None,
                'intra_abdominal': None,
                'liver': None,
                'lung': None,
                'lymph_nodes': None,
                'pleura': None,
                'reproductive_organs': None,
                'other': None,
            },
            'samples': [],
            'diagnoses': [],
            'treatments': [],
            'radiation_therapy': [],
            'surgeries': [],
            'progression_events': [],
            'tumor_site_timeline': [],
            'biomarkers': [],
            'labs': [],
            'clinical_status': [],
        }

    def get_empty_diagnosis_template(self, cancer_type: str) -> Dict[str, Any]:
        """Return empty per-diagnosis/sample structure."""
        return {
            'sample': {
                'sample_id': None,
                'cancer_type': None,
                'cancer_type_detailed': None,
                'oncotree_code': None,
                'sample_type': None,
                'primary_site': None,
                'metastatic_site': None,
                'tumor_purity': None,
                'msi_type': None,
                'msi_score': None,
                'tmb_score': None,
                'tmb_cohort_percentile': None,
                'gene_panel': None,
                'pdl1_positive': None,
                'gleason_sample_level': None,
            },
            'diagnosis': {
                'diagnosis_date': None,
                'dx_description': None,
                'ajcc_stage': None,
                'clinical_stage': None,
                'pathologic_stage': None,
                'stage_cdm_derived': None,
            },
            'treatments': [],
            'radiation_therapy': [],
            'surgeries': [],
            'progression_events': [],
            'biomarkers': [],
        }

    def format_for_prompt(self, cancer_type: str = "generic") -> str:
        """Format MSK-CHORD items for LLM prompts."""
        lines = []
        lines.append(f'Extract data in MSK-CHORD format for: {cancer_type.upper() if cancer_type != "generic" else "All Cancer Types"}')
        lines.append('')
        lines.append('=== PATIENT SUMMARY ===')
        lines.append('Demographics and patient-level clinical attributes:')
        lines.append('- gender: Male/Female')
        lines.append('- race: Race category (self-reported)')
        lines.append('- ethnicity: Ethnicity category (self-reported)')
        lines.append('- current_age: Patient age (capped at 89)')
        lines.append('- ancestry_label: Genomic ancestry (EUR, ASJ, AFR, EAS, SAS, NAM, ADM)')
        lines.append('- smoking_history: Current/Former, Never, or Unknown')
        lines.append('- ecog_kps_first: First ECOG/KPS score (0-4, 4=poorest)')
        lines.append('- bmi_first: First BMI measurement')
        lines.append('- prior_treatment_to_dfci: Prior anti-cancer treatment before comg to DFCI (Yes/No)')
        lines.append('- history_of_pdl1_positive: Ever PD-L1 positive (Yes/No)')
        lines.append('- history_of_dmmr: History of MMR deficiency (Yes/No)')
        lines.append('- gleason_first_reported: First Gleason score (prostate)')
        lines.append('- gleason_highest_reported: Highest Gleason score (prostate)')
        lines.append('- stage_highest_recorded: Highest stage from tumor registry')
        lines.append('- os_status: Living or Deceased')
        lines.append('')
        lines.append('=== TUMOR SITES SUMMARY (Patient-Level NLP-Derived) ===')
        lines.append('Binary flags for tumor sites ever indicated in radiology:')
        lines.append('Sites: Adrenal Glands, Bone, CNS/Brain, Intra-Abdominal, Liver,')
        lines.append('       Lung, Lymph Nodes, Pleura, Reproductive Organs, Other')
        lines.append('')
        lines.append('=== SAMPLE SUMMARY (Per Sequenced Specimen) ===')
        lines.append('- sample_id: Unique sample identifier')
        lines.append('- cancer_type: Main cancer type (Oncotree)')
        lines.append('- cancer_type_detailed: Cancer subtype (Oncotree)')
        lines.append('- oncotree_code: Oncotree classification code')
        lines.append('- sample_type: Primary, Metastasis, Recurrence, Normal')
        lines.append('- primary_site: Primary tumor location')
        lines.append('- metastatic_site: Metastatic site if applicable')
        lines.append('- tumor_purity: Proportion of cancer cells')
        lines.append('- msi_type: Stable, Instable, Indeterminate')
        lines.append('- msi_score: MSI numeric score')
        lines.append('- tmb_score: Tumor Mutational Burden score')
        lines.append('- tmb_cohort_percentile: TMB percentile across all cancers')
        lines.append('- gene_panel: Sequencing panel used (e.g., MSK-IMPACT468)')
        lines.append('- pdl1_positive: Sample PD-L1 positive (Yes/No)')
        lines.append('- gleason_sample_level: Gleason score at sample collection')
        lines.append('')
        lines.append('=== PRIMARY DIAGNOSIS (Tumor Registry) ===')
        lines.append('List of diagnoses from ICD-O tumor registry:')
        lines.append('- diagnosis_date: Date of diagnosis (YYYY-MM-DD)')
        lines.append('- dx_description: ICD-O histology, site, and codes')
        lines.append('- ajcc_stage: AJCC staging')
        lines.append('- clinical_stage: Clinical stage group')
        lines.append('- pathologic_stage: Pathologic stage group')
        lines.append('- stage_cdm_derived: Stage 1-3 or Stage 4')
        lines.append('')
        lines.append('=== ANTI-CANCER MEDICATIONS (Timeline) ===')
        lines.append('List of medications with:')
        lines.append('- start_date, end_date: Treatment duration')
        lines.append('- agent: Drug name (generic)')
        lines.append('- medication_type: Chemo, Hormone, Targeted, Immuno, Biologic, Other')
        lines.append('- investigational: Yes/No for clinical trial drugs')
        lines.append('')
        lines.append('=== RADIATION THERAPY (Timeline) ===')
        lines.append('List of radiation treatments with:')
        lines.append('- start_date, end_date: Treatment dates')
        lines.append('- treatment_site: Anatomic site')
        lines.append('- total_dose: Dose in cGy')
        lines.append('- num_fractions: Number of fractions')
        lines.append('- technique: Radiation technique')
        lines.append('')
        lines.append('=== SURGERY AND IR (Timeline) ===')
        lines.append('List of procedures with:')
        lines.append('- procedure_date: Date performed')
        lines.append('- procedure_type: Surgery or IR')
        lines.append('- description: Procedure description')
        lines.append('- anatomic_site: Site of procedure')
        lines.append('')
        lines.append('=== CANCER PROGRESSION (NLP from Radiology) ===')
        lines.append('List of progression events with:')
        lines.append('- progression_date: Date of assessment')
        lines.append("- progression_status: 'Progressing or Mixed' or 'Improving or Stable'")
        lines.append('- progression_probability: Model probability (0-1)')
        lines.append('- source: CT, MRI, or PET')
        lines.append('')
        lines.append('=== TUMOR SITES (NLP from Radiology Timeline) ===')
        lines.append('List of tumor site indications with:')
        lines.append('- date: Date of radiology report')
        lines.append('- anatomic_location: Site category (9 standard sites)')
        lines.append('')
        lines.append('=== BIOMARKERS (NLP from Pathology) ===')
        lines.append('PD-L1:')
        lines.append('- pdl1_status: Positive/Negative')
        lines.append('- pdl1_score: TPS or CPS if available')
        lines.append('MMR:')
        lines.append('- mmr_status: dMMR (deficient) or pMMR (proficient)')
        lines.append('- mmr_proteins: MLH1, PMS2, MSH2, MSH6 status')
        lines.append('Gleason (prostate):')
        lines.append('- gleason_score: Total Gleason score')
        lines.append('- gleason_primary, gleason_secondary: Pattern scores')
        lines.append('')
        lines.append('=== CANCER ANTIGEN LABS (Timeline) ===')
        lines.append('Tumor markers with date and value:')
        lines.append('- CEA, CA-125, CA-15-3, CA-19-9, PSA, TSH')
        lines.append('')
        lines.append('=== CLINICAL STATUS (Timeline) ===')
        lines.append('- ECOG/KPS scores over time (0-4)')
        lines.append('- Disease-free status windows (min 6 months)')
        lines.append('- BMI measurements over time')
        return '\n'.join(lines)

    def get_supported_cancer_types(self) -> List[str]:
        """List cancer types with Oncotree mappings."""
        return list(ONCOTREE_MAPPINGS.keys())

    def detect_cancer_type(self, primary_site: str = None, histology: str = None, diagnosis_year: int = None) -> str:
        """Map site/oncotree code to cancer type."""
        if not primary_site:
            return 'generic'

        site = primary_site.upper().replace('.', '').replace('-', '').replace('_', '').replace(' ', '')

        # Check against Oncotree mappings
        for cancer_type, codes in ONCOTREE_MAPPINGS.items():
            for code in codes:
                if code in site or site in code:
                    return cancer_type

        # Check ICD-O site codes
        if site.startswith('C34'):
            return 'lung'
        if site.startswith('C50'):
            return 'breast'
        if site.startswith(('C18', 'C19', 'C20')):
            return 'colorectal'
        if site.startswith('C61'):
            return 'prostate'
        if site.startswith('C43'):
            return 'melanoma'
        if site.startswith('C64'):
            return 'kidney'
        if site.startswith('C25'):
            return 'pancreas'
        if site.startswith('C67'):
            return 'bladder'
        if site.startswith('C56'):
            return 'ovarian'
        if site.startswith('C54'):
            return 'endometrial'
        if site.startswith('C71'):
            return 'glioma'
        if site.startswith(('C00', 'C01', 'C02', 'C03', 'C04', 'C05', 'C06', 'C07', 'C08', 'C09', 'C10', 'C11', 'C12', 'C13', 'C14')):
            return 'head_neck'
        if site.startswith('C73'):
            return 'thyroid'
        if site.startswith('C22'):
            return 'liver'
        if site.startswith('C16'):
            return 'gastric'
        if site.startswith('C15'):
            return 'esophageal'

        return 'generic'

    def get_extraction_context(self) -> str:
        """Additional context for MSK-CHORD extraction."""
        return """MSK-CHORD (Clinical Health Oncology Research Data) is the clinical data model
used by Memorial Sloan Kettering Cancer Center for the MSK-IMPACT genomic sequencing study.

Key concepts:
1. Patient Summary: Demographics and aggregate clinical attributes
2. Sample Summary: Per-specimen genomic and clinical data (Oncotree cancer types)
3. Timeline Data: Longitudinal clinical events with dates

Data sources:
- Tumor Registry (ICD-O): Diagnoses and staging
- EMR: Treatments, procedures, labs
- NLP-derived: Smoking, progression, tumor sites, biomarkers (PD-L1, MMR, Gleason)

Important notes:
- Dates should be in YYYY-MM-DD format
- Use Oncotree codes for cancer classification
- NLP-derived fields have specific confidence/probability values
- MSI/TMB are sample-level genomic features
- Timeline data represents longitudinal clinical events"""
