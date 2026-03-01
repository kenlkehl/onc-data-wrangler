"""
MatchMiner-AI Ontology Implementation

Implements OntologyBase for clinical trial matching and disease characterization.
Provides structured extraction of concepts relevant to clinical trial eligibility.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any

from ...base import OntologyBase, DataItem, DataCategory


# Directory containing this module
MODULE_DIR = Path(__file__).parent


def _load_json_file(filepath: Path) -> Dict:
    """Load a JSON file and return its contents as a dictionary."""
    if not filepath.exists():
        return {}
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


CANCER_TYPE_BIOMARKERS = {
    'nsclc': ['EGFR', 'ALK', 'ROS1', 'BRAF', 'KRAS', 'MET', 'RET', 'NTRK', 'HER2', 'PD-L1'],
    'lung': ['EGFR', 'ALK', 'ROS1', 'BRAF', 'KRAS', 'MET', 'RET', 'NTRK', 'HER2', 'PD-L1'],
    'breast': ['ER', 'PR', 'HER2', 'BRCA1', 'BRCA2', 'PIK3CA', 'ESR1', 'Ki-67'],
    'colorectal': ['KRAS', 'NRAS', 'BRAF', 'MSI', 'HER2', 'NTRK'],
    'prostate': ['AR', 'BRCA1', 'BRCA2', 'ATM', 'PALB2', 'CDK12', 'MSI'],
    'melanoma': ['BRAF', 'NRAS', 'KIT', 'PD-L1'],
    'ovarian': ['BRCA1', 'BRCA2', 'HRD'],
    'pancreatic': ['KRAS', 'BRCA1', 'BRCA2', 'MSI', 'NTRK'],
    'gastric': ['HER2', 'MSI', 'PD-L1', 'Claudin 18.2'],
    'renal': ['PD-L1'],
    'bladder': ['FGFR', 'PD-L1', 'HER2'],
    'thyroid': ['BRAF', 'RET', 'NTRK'],
    'generic': ['MSI', 'TMB', 'PD-L1', 'NTRK'],
}

RISK_SYSTEMS = {
    'prostate': 'NCCN Risk (Very Low/Low/Favorable Intermediate/Unfavorable Intermediate/High/Very High)',
    'kidney': 'IMDC Risk (Favorable/Intermediate/Poor)',
    'renal': 'IMDC Risk (Favorable/Intermediate/Poor)',
    'breast': 'Oncotype DX / MammaPrint / Risk Category',
    'myeloma': 'ISS / R-ISS Stage',
    'leukemia': 'ELN Risk (Favorable/Intermediate/Adverse)',
    'mds': 'IPSS-R (Very Low/Low/Intermediate/High/Very High)',
    'lymphoma': 'IPI / FLIPI / MIPI',
    'generic': 'Cancer-specific risk stratification if applicable',
}


class MatchMinerAIOntology(OntologyBase):
    """Ontology for clinical trial matching based on MatchMiner-AI concepts."""

    @property
    def ontology_id(self) -> str:
        return 'matchminer_ai'

    @property
    def display_name(self) -> str:
        return 'MatchMiner-AI Clinical Trial Matching'

    @property
    def version(self) -> str:
        return '1.0.0'

    @property
    def description(self) -> str:
        return 'Clinical trial matching concepts'

    def __init__(self):
        """Initialize and load concept definitions."""
        self._concepts = self._load_all_concepts()

    def _load_all_concepts(self) -> dict:
        """Load all concept JSON files."""
        concepts_dir = MODULE_DIR / 'concepts'
        concepts = {}
        for filename in ('patient_level.json', 'per_diagnosis.json', 'clinical_trial_eligibility.json', 'disease_status.json', 'comorbidities.json', 'longitudinal_treatment.json'):
            filepath = concepts_dir / filename
            if not filepath.exists():
                continue
            data = _load_json_file(filepath)
            concepts[data.get('category', filename.replace('.json', ''))] = data
        return concepts

    def get_base_items(self) -> List[DataCategory]:
        """Get patient-level data categories."""
        categories = []

        # Patient-level items
        if 'patient_level' in self._concepts:
            data = self._concepts['patient_level']
            items = []
            for item_data in data.get('items', []):
                items.append(DataItem(
                    id=item_data.get('id', ''),
                    name=item_data.get('name', ''),
                    description=item_data.get('description', ''),
                    data_type=item_data.get('data_type', 'string'),
                    valid_values=item_data.get('valid_values'),
                    extraction_hints=item_data.get('extraction_hints', []),
                    json_field=item_data.get('json_field'),
                ))
            categories.append(DataCategory(
                id='patient_level',
                name='Patient Demographics',
                description=data.get('description', ''),
                items=items,
                per_diagnosis=False,
            ))

        # Disease status items
        if 'disease_status' in self._concepts:
            data = self._concepts['disease_status']
            items = []
            for item_data in data.get('items', []):
                items.append(DataItem(
                    id=item_data.get('id', ''),
                    name=item_data.get('name', ''),
                    description=item_data.get('description', ''),
                    data_type=item_data.get('data_type', 'object'),
                    extraction_hints=item_data.get('extraction_hints', []),
                    json_field=item_data.get('json_field'),
                ))
            categories.append(DataCategory(
                id='disease_status',
                name='Disease Status',
                description=data.get('description', ''),
                items=items,
                per_diagnosis=False,
            ))

        # Clinical trial eligibility items
        if 'clinical_trial_eligibility' in self._concepts:
            data = self._concepts['clinical_trial_eligibility']
            items = []
            for item_data in data.get('items', []):
                items.append(DataItem(
                    id=item_data.get('id', ''),
                    name=item_data.get('name', ''),
                    description=item_data.get('description', ''),
                    data_type=item_data.get('data_type', 'object'),
                    extraction_hints=item_data.get('extraction_hints', []),
                    json_field=item_data.get('json_field'),
                ))
            categories.append(DataCategory(
                id='clinical_trial_eligibility',
                name='Clinical Trial Eligibility Concerns',
                description=data.get('description', ''),
                items=items,
                per_diagnosis=False,
            ))

        # Comorbidities items
        if 'comorbidities' in self._concepts:
            data = self._concepts['comorbidities']
            items = []
            for item_data in data.get('items', []):
                items.append(DataItem(
                    id=item_data.get('id', ''),
                    name=item_data.get('name', ''),
                    description=item_data.get('description', ''),
                    data_type=item_data.get('data_type', 'object'),
                    extraction_hints=item_data.get('extraction_hints', []),
                    json_field=item_data.get('json_field'),
                ))
            categories.append(DataCategory(
                id='comorbidities',
                name='Comorbidities',
                description=data.get('description', ''),
                items=items,
                per_diagnosis=False,
            ))

        return categories

    def get_site_specific_items(self, cancer_type: str) -> List[DataCategory]:
        """Get cancer-type-specific items (primarily biomarkers)."""
        categories = []

        if 'per_diagnosis' in self._concepts:
            data = self._concepts['per_diagnosis']
            items = []
            for item_data in data.get('items', []):
                items.append(DataItem(
                    id=item_data.get('id', ''),
                    name=item_data.get('name', ''),
                    description=item_data.get('description', ''),
                    data_type=item_data.get('data_type', 'string'),
                    extraction_hints=item_data.get('extraction_hints', []),
                    json_field=item_data.get('json_field'),
                ))
            categories.append(DataCategory(
                id='per_diagnosis',
                name='Per-Diagnosis Concepts',
                description=data.get('description', ''),
                items=items,
                per_diagnosis=True,
                context=f'Key biomarkers for {cancer_type}: {", ".join(CANCER_TYPE_BIOMARKERS.get(cancer_type.lower(), CANCER_TYPE_BIOMARKERS["generic"]))}',
            ))

        return categories

    def get_empty_summary_template(self) -> Dict[str, Any]:
        """Return empty MatchMiner-AI JSON structure."""
        return {
            'version': self.version,
            'patient': {
                'age': None,
                'sex': None,
                'ecog_performance_status': None,
            },
            'diagnoses': [],
            'current_disease_status': {
                'as_of_date': None,
                'status': None,
                'evidence_basis': None,
            },
            'clinical_trial_eligibility_concerns': {
                'brain_metastases': {
                    'present': None,
                    'status': None,
                },
                'measurable_disease': {
                    'present': None,
                },
                'organ_function': {},
                'prior_immunotherapy': {
                    'received': None,
                    'agents': [],
                    'immune_adverse_events': [],
                },
                'autoimmune_conditions': [],
                'recent_surgery': {},
                'active_infections': {},
            },
            'comorbidities': {
                'cardiac': [],
                'pulmonary': [],
                'renal': [],
                'hepatic': [],
                'endocrine': [],
                'neurologic': [],
                'hematologic': [],
                'gastrointestinal': [],
                'psychiatric': [],
                'rheumatologic': [],
                'infectious': [],
                'other': [],
            },
            'treatment_history': [],
            'current_treatment': None,
        }

    def get_empty_diagnosis_template(self, cancer_type: str) -> Dict[str, Any]:
        """Return empty per-diagnosis structure."""
        biomarkers_template = {}
        for marker in CANCER_TYPE_BIOMARKERS.get(cancer_type.lower(), CANCER_TYPE_BIOMARKERS['generic']):
            biomarkers_template[marker] = {'status': None}

        return {
            'diagnosis_index': None,
            'cancer_type': None,
            'histology': None,
            'extent_of_disease': {
                'intent': None,
                'stage_category': None,
                'sites_of_disease': [],
            },
            'risk_category': None,
            'biomarkers': biomarkers_template,
            'prior_treatments': [],
        }

    def format_for_prompt(self, cancer_type: str = "generic") -> str:
        """Format MatchMiner-AI items for LLM prompts."""
        lines = []

        lines.append('Extract clinical trial matching concepts:')
        lines.append('')

        # PATIENT-LEVEL section
        lines.append('=== PATIENT-LEVEL ===')
        lines.append('- Age: Current age in years')
        lines.append('- Sex: Male/Female')
        lines.append('- ECOG Performance Status: 0-4 scale')
        lines.append('')

        # PER-DIAGNOSIS section
        lines.append('=== PER-DIAGNOSIS (list of dicts, one per cancer diagnosis) ===')
        lines.append("- cancer_type: Specific cancer type (e.g., 'Non-Small Cell Lung Cancer')")
        lines.append('- histology: Histologic subtype')
        lines.append('- extent_of_disease:')
        lines.append("    - intent: 'Curative' or 'Palliative'")
        lines.append("    - stage_category: 'Early-stage', 'Locally Advanced', or 'Metastatic'")
        lines.append('    - sites_of_disease: Array of current disease sites')
        risk_system = RISK_SYSTEMS.get(cancer_type.lower(), RISK_SYSTEMS['generic'])
        lines.append(f'- risk_category: {risk_system}')

        biomarkers = CANCER_TYPE_BIOMARKERS.get(cancer_type.lower(), CANCER_TYPE_BIOMARKERS['generic'])
        lines.append(f'- biomarkers: Key markers for {cancer_type}: {", ".join(biomarkers)}')
        lines.append('    Each biomarker should have: status, variant (if applicable)')
        lines.append('- prior_treatments: Array of prior systemic treatments for this diagnosis')
        lines.append('    Each treatment: regimen, line_of_therapy, intent, best_response, ongoing')
        lines.append('')

        # CURRENT DISEASE STATUS section
        lines.append('=== CURRENT DISEASE STATUS ===')
        lines.append('- as_of_date: Date of most recent assessment')
        lines.append('- status: One of:')
        lines.append("    'No evidence of disease (NED)', 'Stable disease',")
        lines.append("    'Responding to treatment', 'Progressing on treatment',")
        lines.append("    'Metastatic, on active treatment', 'Newly diagnosed, not yet treated'")
        lines.append("- evidence_basis: 'Imaging', 'Labs/Tumor Markers', 'Clinical', 'Pathology'")
        lines.append('')

        # CLINICAL TRIAL ELIGIBILITY CONCERNS section
        lines.append('=== CLINICAL TRIAL ELIGIBILITY CONCERNS ===')
        lines.append('- brain_metastases: present (bool), status (Active/Treated stable/Leptomeningeal)')
        lines.append('- measurable_disease: present (bool) per RECIST 1.1')
        lines.append('- organ_function: renal (Cr, CrCl), hepatic (AST, ALT, bili), hematologic (ANC, plt, Hgb)')
        lines.append('- prior_immunotherapy: received (bool), agents [], immune_adverse_events []')
        lines.append('- autoimmune_conditions: array of autoimmune diseases')
        lines.append('- recent_surgery: major_surgery_date, procedure_type, fully_healed')
        lines.append('- active_infections: HIV, HBV, HCV status, other infections')
        lines.append('')

        # COMORBIDITIES section
        lines.append('=== COMORBIDITIES (by organ system) ===')
        lines.append('- cardiac: heart conditions (CHF, afib, CAD, etc.)')
        lines.append('- pulmonary: lung conditions (COPD, ILD, etc.)')
        lines.append('- renal: kidney conditions')
        lines.append('- hepatic: liver conditions')
        lines.append('- endocrine: diabetes, thyroid, etc.')
        lines.append('- neurologic: stroke, seizures, neuropathy, etc.')
        lines.append('- rheumatologic: autoimmune conditions')
        lines.append('- other: additional significant conditions')
        lines.append('')

        # TREATMENT HISTORY section
        lines.append('=== TREATMENT HISTORY ===')
        lines.append('- treatment_history: Array of all cancer treatments')
        lines.append('    Each: line_of_therapy, regimen_name, drugs [], drug_classes [],')
        lines.append('    intent, start_date, end_date, ongoing, cycles_completed,')
        lines.append('    best_response, reason_for_discontinuation, adverse_events []')
        lines.append('- current_treatment: Details of ongoing treatment')
        lines.append('- radiation_history: site, modality, dose, fractions, completion_date')
        lines.append('- surgical_history: procedure, date, intent, margins')
        lines.append('')

        return '\n'.join(lines)

    def get_supported_cancer_types(self) -> List[str]:
        """List cancer types with specific biomarker panels."""
        return list(CANCER_TYPE_BIOMARKERS.keys())

    def detect_cancer_type(self, primary_site: str = None, histology: str = None, diagnosis_year: int = None) -> str:
        """Map to cancer type for biomarker panel selection."""
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
        if site.startswith('C44'):
            return 'melanoma'
        if site.startswith('C56'):
            return 'ovarian'
        if site.startswith('C25'):
            return 'pancreatic'
        if site.startswith('C16'):
            return 'gastric'
        if site.startswith('C64'):
            return 'renal'
        if site.startswith('C67'):
            return 'bladder'
        if site.startswith('C73'):
            return 'thyroid'

        return 'generic'

    def get_extraction_context(self) -> str:
        """Additional context for MatchMiner-AI extraction."""
        return (
            "MatchMiner-AI concepts are designed for clinical trial matching.\n"
            "Focus on extracting:\n"
            "1. Current disease status and extent (metastatic vs localized, treatment intent)\n"
            "2. Actionable biomarkers relevant to the cancer type\n"
            "3. Prior treatments and responses (all lines of therapy)\n"
            "4. Common trial exclusion criteria (brain mets, organ function, infections)\n"
            "5. Comorbidities that may affect eligibility\n"
            "\n"
            "For patients with multiple cancer diagnoses, create separate entries in the\n"
            "'diagnoses' array with each diagnosis's specific characteristics."
        )
