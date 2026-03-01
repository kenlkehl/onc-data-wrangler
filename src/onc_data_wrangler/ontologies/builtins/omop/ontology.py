"""
OMOP CDM Ontology Implementation

Implements OntologyBase for OMOP Common Data Model v5.4 with Oncology Extension.
Provides structured extraction aligned with OHDSI standards.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any

from ...base import OntologyBase, DataItem, DataCategory


MODULE_DIR = Path(__file__).parent


def _load_json_file(filepath: Path) -> Dict:
    """Load a JSON file and return its contents."""
    if not filepath.exists():
        return {}
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


EPISODE_CONCEPTS = {
    'disease_first_occurrence': {'id': 32528, 'name': 'Disease First Occurrence'},
    'disease_recurrence': {'id': 32545, 'name': 'Disease Recurrence'},
    'disease_progression': {'id': 32530, 'name': 'Disease Progression'},
    'disease_remission': {'id': 32546, 'name': 'Disease Remission'},
    'treatment_regimen': {'id': 32531, 'name': 'Treatment Regimen'},
    'treatment_cycle': {'id': 32532, 'name': 'Treatment Cycle'},
}

CANCER_SNOMED_CONCEPTS = {
    'lung': {'id': 254637007, 'name': 'Non-small cell lung cancer'},
    'nsclc': {'id': 254637007, 'name': 'Non-small cell lung cancer'},
    'sclc': {'id': 254632001, 'name': 'Small cell lung cancer'},
    'breast': {'id': 254838004, 'name': 'Carcinoma of breast'},
    'colorectal': {'id': 363406005, 'name': 'Colorectal cancer'},
    'prostate': {'id': 399068003, 'name': 'Prostate cancer'},
    'melanoma': {'id': 372244006, 'name': 'Malignant melanoma'},
    'kidney': {'id': 93849006, 'name': 'Renal cell carcinoma'},
    'pancreas': {'id': 372142002, 'name': 'Pancreatic carcinoma'},
    'generic': {'id': 363346000, 'name': 'Malignant neoplastic disease'},
}


class OMOPOntology(OntologyBase):
    """
    OMOP CDM v5.4 + Oncology Extension Ontology.

    Provides data extraction aligned with OHDSI OMOP Common Data Model,
    including the Episode domain for oncology.
    """

    def __init__(self):
        """Initialize and load table definitions."""
        self._vocabularies = self._load_vocabularies()
        self._oncology_ext = self._load_oncology_extension()

    def _load_vocabularies(self) -> dict:
        """Load vocabulary definitions."""
        vocab_dir = MODULE_DIR / 'vocabularies'
        data = {}
        for filename in ('condition_occurrence.json', 'drug_exposure.json', 'procedure_occurrence.json', 'measurement.json'):
            filepath = vocab_dir / filename
            if not filepath.exists():
                continue
            table_name = filename.replace('.json', '')
            data[table_name] = _load_json_file(filepath)
        return data

    def _load_oncology_extension(self) -> dict:
        """Load oncology extension definitions."""
        ext_dir = MODULE_DIR / 'oncology_extension'
        data = {}
        for filename in ('episode.json', 'episode_event.json', 'cancer_modifiers.json'):
            filepath = ext_dir / filename
            if not filepath.exists():
                continue
            table_name = filename.replace('.json', '')
            data[table_name] = _load_json_file(filepath)
        return data

    @property
    def ontology_id(self) -> str:
        return 'omop'

    @property
    def display_name(self) -> str:
        return 'OMOP CDM v5.4 + Oncology Extension'

    @property
    def version(self) -> str:
        return '5.4.0'

    @property
    def description(self) -> str:
        return 'OMOP Common Data Model oncology extension'

    def get_base_items(self) -> List[DataCategory]:
        """Get person-level OMOP items."""
        categories = []
        person_items = [
            DataItem(id='person_id', name='Person ID', description='Unique patient identifier', data_type='integer'),
            DataItem(id='year_of_birth', name='Year of Birth', description='Birth year', data_type='integer'),
            DataItem(id='month_of_birth', name='Month of Birth', description='Birth month', data_type='integer'),
            DataItem(id='day_of_birth', name='Day of Birth', description='Birth day', data_type='integer'),
            DataItem(id='gender_concept_id', name='Gender Concept ID', description='OMOP gender concept', data_type='integer'),
            DataItem(id='gender_concept_name', name='Gender', description='Gender (Male/Female)', data_type='string'),
            DataItem(id='race_concept_id', name='Race Concept ID', description='OMOP race concept', data_type='integer'),
            DataItem(id='race_concept_name', name='Race', description='Race category', data_type='string'),
            DataItem(id='ethnicity_concept_id', name='Ethnicity Concept ID', description='OMOP ethnicity concept', data_type='integer'),
            DataItem(id='ethnicity_concept_name', name='Ethnicity', description='Ethnicity category', data_type='string'),
        ]
        categories.append(DataCategory(id='person', name='Person', description='Patient demographics', items=person_items, per_diagnosis=False))
        return categories

    def get_site_specific_items(self, cancer_type: str) -> List[DataCategory]:
        """Get condition, drug, procedure, measurement, and episode items."""
        categories = []

        cond_items = [
            DataItem(id='condition_concept_id', name='Condition Concept ID', description='SNOMED/ICD-O-3 concept', data_type='integer'),
            DataItem(id='condition_concept_name', name='Condition Name', description='Cancer diagnosis', data_type='string'),
            DataItem(id='condition_start_date', name='Condition Start Date', description='Diagnosis date', data_type='date'),
            DataItem(id='condition_source_value', name='Condition Source Value', description='ICD-10/ICD-O-3 code', data_type='string'),
            DataItem(id='topography_concept_id', name='Topography Concept ID', description='ICD-O-3 site code', data_type='integer'),
            DataItem(id='topography_source_value', name='Topography', description='Primary site (e.g., C34.1)', data_type='string'),
            DataItem(id='morphology_concept_id', name='Morphology Concept ID', description='ICD-O-3 histology code', data_type='integer'),
            DataItem(id='morphology_source_value', name='Morphology', description='Histology (e.g., 8140/3)', data_type='string'),
        ]
        categories.append(DataCategory(id='condition_occurrence', name='Condition Occurrence', description='Cancer diagnoses', items=cond_items, per_diagnosis=True))

        episode_items = [
            DataItem(id='episode_id', name='Episode ID', description='Unique episode identifier', data_type='integer'),
            DataItem(id='episode_concept_id', name='Episode Concept ID', description='Episode type concept', data_type='integer'),
            DataItem(id='episode_concept_name', name='Episode Type', description='Disease First Occurrence, Treatment Regimen, etc.', data_type='string'),
            DataItem(id='episode_start_date', name='Episode Start Date', description='Episode begin date', data_type='date'),
            DataItem(id='episode_end_date', name='Episode End Date', description='Episode end date', data_type='date'),
            DataItem(id='episode_object_concept_id', name='Episode Object Concept ID', description='What the episode is about', data_type='integer'),
            DataItem(id='episode_object_concept_name', name='Episode Object', description='Cancer type or regimen name', data_type='string'),
        ]
        categories.append(DataCategory(id='episode', name='Episode (Oncology Extension)', description='Disease and treatment episodes', items=episode_items, per_diagnosis=True))

        drug_items = [
            DataItem(id='drug_concept_id', name='Drug Concept ID', description='RxNorm/HemOnc concept', data_type='integer'),
            DataItem(id='drug_concept_name', name='Drug Name', description='Treatment name', data_type='string'),
            DataItem(id='drug_exposure_start_date', name='Start Date', description='Treatment start', data_type='date'),
            DataItem(id='drug_exposure_end_date', name='End Date', description='Treatment end', data_type='date'),
            DataItem(id='route_concept_name', name='Route', description='Administration route (IV, PO, etc.)', data_type='string'),
            DataItem(id='drug_source_value', name='Source Drug Name', description='Original drug name', data_type='string'),
        ]
        categories.append(DataCategory(id='drug_exposure', name='Drug Exposure', description='Cancer treatments (chemotherapy, immunotherapy, targeted therapy)', items=drug_items, per_diagnosis=True))

        proc_items = [
            DataItem(id='procedure_concept_id', name='Procedure Concept ID', description='SNOMED/CPT concept', data_type='integer'),
            DataItem(id='procedure_concept_name', name='Procedure Name', description='Procedure description', data_type='string'),
            DataItem(id='procedure_date', name='Procedure Date', description='Date performed', data_type='date'),
            DataItem(id='procedure_source_value', name='Source Value', description='CPT/HCPCS code', data_type='string'),
        ]
        categories.append(DataCategory(id='procedure_occurrence', name='Procedure Occurrence', description='Surgical and diagnostic procedures', items=proc_items, per_diagnosis=True))

        meas_items = [
            DataItem(id='measurement_concept_id', name='Measurement Concept ID', description='LOINC concept', data_type='integer'),
            DataItem(id='measurement_concept_name', name='Measurement Name', description='Lab/biomarker name', data_type='string'),
            DataItem(id='measurement_date', name='Measurement Date', description='Date of measurement', data_type='date'),
            DataItem(id='value_as_number', name='Numeric Value', description='Numeric result', data_type='number'),
            DataItem(id='value_as_concept_name', name='Categorical Value', description='Categorical result (Positive/Negative)', data_type='string'),
            DataItem(id='unit_concept_name', name='Unit', description='Unit of measure', data_type='string'),
        ]
        categories.append(DataCategory(id='measurement', name='Measurement', description='Labs, tumor markers, biomarkers', items=meas_items, per_diagnosis=True))

        return categories

    def get_empty_summary_template(self) -> Dict[str, Any]:
        """Return empty OMOP JSON structure."""
        return {
            'version': self.version,
            'person': {
                'person_id': None,
                'year_of_birth': None,
                'month_of_birth': None,
                'day_of_birth': None,
                'gender_concept_id': None,
                'gender_concept_name': None,
                'race_concept_id': None,
                'race_concept_name': None,
                'ethnicity_concept_id': None,
                'ethnicity_concept_name': None,
            },
            'condition_occurrences': [],
            'episodes': [],
            'drug_exposures': [],
            'procedure_occurrences': [],
            'measurements': [],
        }

    def get_empty_diagnosis_template(self, cancer_type: str) -> Dict[str, Any]:
        """Return empty per-condition structure."""
        cancer_concept = CANCER_SNOMED_CONCEPTS.get(cancer_type.lower(), CANCER_SNOMED_CONCEPTS['generic'])

        return {
            'condition_occurrence': {
                'condition_concept_id': cancer_concept['id'],
                'condition_concept_name': cancer_concept['name'],
                'condition_start_date': None,
                'condition_source_value': None,
                'topography_source_value': None,
                'morphology_source_value': None,
            },
            'disease_episode': {
                'episode_concept_id': EPISODE_CONCEPTS['disease_first_occurrence']['id'],
                'episode_concept_name': EPISODE_CONCEPTS['disease_first_occurrence']['name'],
                'episode_start_date': None,
                'episode_end_date': None,
                'episode_object_concept_id': cancer_concept['id'],
                'episode_object_concept_name': cancer_concept['name'],
                'disease_extent': None,
                'disease_status': None,
            },
            'treatment_episodes': [],
            'drug_exposures': [],
            'procedure_occurrences': [],
            'measurements': [],
        }

    def format_for_prompt(self, cancer_type: str = "generic") -> str:
        """Format OMOP items for LLM prompts."""
        lines = []
        cancer_concept = CANCER_SNOMED_CONCEPTS.get(cancer_type.lower(), CANCER_SNOMED_CONCEPTS['generic'])

        lines.append(f'Extract data in OMOP CDM v5.4 format for: {cancer_concept["name"]}')
        lines.append('')

        # PERSON
        lines.append('=== PERSON (Patient Demographics) ===')
        lines.append('- year_of_birth, month_of_birth, day_of_birth')
        lines.append('- gender_concept_name: Male/Female')
        lines.append('- race_concept_name: White/Black/Asian/Other')
        lines.append('- ethnicity_concept_name: Hispanic or Latino / Not Hispanic or Latino')
        lines.append('')

        # CONDITION_OCCURRENCE
        lines.append('=== CONDITION_OCCURRENCE (Cancer Diagnoses) ===')
        lines.append(f"- condition_concept_name: e.g., '{cancer_concept['name']}'")
        lines.append('- condition_start_date: Diagnosis date (YYYY-MM-DD)')
        lines.append('- condition_source_value: Source ICD-10/ICD-O-3 code')
        lines.append('- topography_source_value: ICD-O-3 site (e.g., C34.1)')
        lines.append('- morphology_source_value: ICD-O-3 histology (e.g., 8140/3)')
        lines.append('')

        # EPISODE - Disease Episodes
        lines.append('=== EPISODE (Oncology Extension - Disease Episodes) ===')
        lines.append('Episode types:')
        lines.append('- Disease First Occurrence: Initial cancer diagnosis')
        lines.append('- Disease Recurrence: Cancer recurrence after treatment')
        lines.append('- Disease Progression: Progression on therapy')
        lines.append('- Disease Remission: Complete or partial remission')
        lines.append('')
        lines.append('Each disease episode includes:')
        lines.append('- episode_concept_name: Episode type from above')
        lines.append('- episode_start_date, episode_end_date')
        lines.append('- episode_object_concept_name: Cancer type')
        lines.append('- disease_extent: Localized/Regional/Distant')
        lines.append('- disease_status: Stable/Responding/Progressing')
        lines.append('')

        # EPISODE - Treatment Episodes
        lines.append('=== EPISODE (Oncology Extension - Treatment Episodes) ===')
        lines.append('Episode types:')
        lines.append('- Treatment Regimen: A course of cancer-directed therapy')
        lines.append('')
        lines.append('Each treatment episode includes:')
        lines.append("- episode_concept_name: 'Treatment Regimen'")
        lines.append('- episode_start_date, episode_end_date')
        lines.append('- episode_object_concept_name: Regimen name (HemOnc)')
        lines.append("  Examples: 'Carboplatin/Pemetrexed/Pembrolizumab', 'FOLFOX', 'Osimertinib'")
        lines.append('- treatment_setting: Neoadjuvant/Adjuvant/First-line/Second-line+')
        lines.append('- best_response: CR/PR/SD/PD')
        lines.append('')

        # DRUG_EXPOSURE
        lines.append('=== DRUG_EXPOSURE (Individual Drugs) ===')
        lines.append('- drug_concept_name: Drug name (RxNorm standard)')
        lines.append('- drug_exposure_start_date, drug_exposure_end_date')
        lines.append('- route_concept_name: IV/PO/Subcutaneous')
        lines.append('- drug_source_value: Original drug name from source')
        lines.append('')

        # PROCEDURE_OCCURRENCE
        lines.append('=== PROCEDURE_OCCURRENCE ===')
        lines.append('- procedure_concept_name: Procedure description')
        lines.append('  Examples: Lobectomy, Mastectomy, SBRT, Biopsy')
        lines.append('- procedure_date: Date performed')
        lines.append('- procedure_source_value: CPT/HCPCS code if available')
        lines.append('')

        # MEASUREMENT
        lines.append('=== MEASUREMENT (Labs, Biomarkers) ===')
        lines.append('- measurement_concept_name: LOINC name')
        lines.append('- measurement_date: Date of measurement')
        lines.append('- value_as_number: Numeric result')
        lines.append('- value_as_concept_name: Categorical result (Positive/Negative)')
        lines.append('- unit_concept_name: Unit (%, ng/mL, etc.)')
        lines.append('')
        lines.append('Common oncology measurements:')
        lines.append('- Tumor markers: PSA, CEA, CA-125, CA 19-9, AFP')
        lines.append('- Genomic: EGFR, ALK, KRAS, BRAF, MSI, TMB, PD-L1')
        lines.append('- Receptor status: ER, PR, HER2')
        lines.append('')

        return '\n'.join(lines)

    def get_supported_cancer_types(self) -> List[str]:
        """List cancer types with SNOMED concept mappings."""
        return list(CANCER_SNOMED_CONCEPTS.keys())

    def detect_cancer_type(self, primary_site: str = None, histology: str = None, diagnosis_year: int = None) -> str:
        """Map site to cancer type for SNOMED concept selection."""
        if not primary_site:
            return 'generic'

        site = primary_site.upper().replace('.', '')

        if site.startswith('C34'):
            if histology and histology[:4] in ('8041', '8042', '8043', '8044', '8045'):
                return 'sclc'
            return 'nsclc'
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

        return 'generic'

    def get_extraction_context(self) -> str:
        """Additional context for OMOP extraction."""
        return """OMOP CDM (Observational Medical Outcomes Partnership Common Data Model) is the
standard data model used by OHDSI for observational health research.

Key concepts:
1. Use concept IDs where possible, but also provide human-readable names
2. Episodes aggregate clinical events into meaningful disease phases and treatment courses
3. Disease Episodes track: First Occurrence, Recurrence, Progression, Remission
4. Treatment Episodes track regimens (linked to drug_exposures via episode_event)
5. Use HemOnc vocabulary for regimen names (e.g., 'FOLFOX', 'AC-T')
6. Dates should be in YYYY-MM-DD format

The oncology extension enables sophisticated cancer research by connecting
low-level clinical events to high-level disease and treatment abstractions."""
