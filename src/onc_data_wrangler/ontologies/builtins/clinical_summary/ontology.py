"""
Clinical Summary Ontology Implementation

Produces free-text clinical summaries rather than structured JSON extractions.
Adapted from MatchMiner-AI serial summarization prompts. The summary captures
key oncology information (diagnosis, biomarkers, treatments, eligibility
concerns) in a concise narrative format suitable for downstream structured
extraction or clinical trial matching.
"""

from typing import Dict, List, Any

from ...base import OntologyBase, DataItem, DataCategory


class ClinicalSummaryOntology(OntologyBase):
    """Ontology that produces free-text clinical summaries."""

    @property
    def is_free_text(self) -> bool:
        return True

    @property
    def ontology_id(self) -> str:
        return 'clinical_summary'

    @property
    def display_name(self) -> str:
        return 'Clinical Summary (Free Text)'

    @property
    def version(self) -> str:
        return '1.0.0'

    @property
    def description(self) -> str:
        return 'Free-text clinical oncology summary (diagnosis, biomarkers, treatment, eligibility)'

    def get_base_items(self) -> List[DataCategory]:
        """Advisory list of information that should appear in the summary."""
        items = [
            DataItem(id='age', name='Age', description='Current age in years', data_type='string'),
            DataItem(id='sex', name='Sex', description='Patient sex', data_type='string'),
            DataItem(id='cancer_type', name='Cancer Type', description='Cancer type / primary site', data_type='string'),
            DataItem(id='histology', name='Histology', description='Histologic subtype', data_type='string'),
            DataItem(id='extent', name='Current Extent', description='Localized, advanced, metastatic, etc.', data_type='string'),
            DataItem(id='biomarkers', name='Biomarkers', description='Genomic results, protein expression, etc.', data_type='string'),
            DataItem(id='treatment_history', name='Treatment History', description='Surgery, radiation, systemic therapy with dates and responses', data_type='string'),
            DataItem(id='boilerplate', name='Boilerplate Exclusion Criteria', description='Brain mets, organ dysfunction, infections, etc.', data_type='string'),
        ]
        return [DataCategory(
            id='clinical_summary',
            name='Clinical Summary',
            description='Free-text summary of oncology history',
            items=items,
            per_diagnosis=False,
        )]

    def get_site_specific_items(self, cancer_type: str) -> List[DataCategory]:
        """No site-specific items for free-text summary."""
        return []

    def get_empty_summary_template(self) -> Dict[str, Any]:
        return {'summary': ''}

    def get_empty_diagnosis_template(self, cancer_type: str) -> Dict[str, Any]:
        return {'summary': ''}

    def format_for_prompt(self, cancer_type: str = "generic") -> str:
        """Return the free-text summarization prompt."""
        return SUMMARY_PROMPT

    def get_supported_cancer_types(self) -> List[str]:
        return ['generic']

    def detect_cancer_type(self, primary_site: str = None, histology: str = None, diagnosis_year: int = None) -> str:
        return 'generic'

    def get_extraction_context(self) -> str:
        return (
            "This ontology produces a free-text clinical summary rather than "
            "structured JSON. The summary should capture all clinically relevant "
            "oncology information in a concise narrative format."
        )

    def validate_output(self, output: Dict[str, Any]) -> List[str]:
        errors = []
        if not isinstance(output.get('summary', ''), str):
            errors.append("'summary' must be a string")
        return errors


SUMMARY_PROMPT = """\
You are an experienced clinical oncology history summarization bot.

You are summarizing a patient's cancer history based on their electronic health record.

Your task:
- Write a concise, comprehensive summary of the patient's oncology history.
- If the patient does not yet have a cancer diagnosis, state "No cancer diagnosis documented" and summarize relevant medical history.

Document the following:
- Age (most recent)
- Sex
- Cancer type / primary site (eg breast cancer, lung cancer)
- Histology (eg adenocarcinoma, squamous carcinoma)
- Current extent (localized, advanced, metastatic, etc)
- Stage at diagnosis if known
- Biomarkers (genomic results, protein expression, etc)
- Treatment history (surgery, radiation, chemotherapy/targeted therapy/immunotherapy, including start/stop dates and best response if known)

Do not consider localized basal cell or squamous carcinomas of the skin, or colon polyps, to be cancers.
Do not include the patient's name, but do include relevant dates whenever documented.
If a patient has more than one cancer, document the cancers one at a time. List the currently or most recently active cancer first, followed by prior cancers. Within each cancer, events should be in chronological order.

Also document any history of conditions that might meet "boilerplate" exclusion criteria for clinical trials, including:
- Uncontrolled brain metastases
- Lack of measurable disease
- Congestive heart failure
- Pneumonitis
- Renal dysfunction
- Liver dysfunction
- HIV or hepatitis infection

Clearly separate the "boilerplate" section by labeling it "Boilerplate:" before describing any such conditions.

CRITICAL: Format your response as free text ONLY. Do NOT output markdown, JSON, Unicode, or tables.

Here is an example of the desired output format:

Age: 70
Sex: Male
Cancer type: Lung cancer
Histology: Adenocarcinoma
Stage at diagnosis: IIIB
Current extent: Metastatic
Biomarkers: PD-L1 75%, KRAS G12C mutant
Treatment history:
# 1/5/2020-2/5/2021: carboplatin/pemetrexed/pembrolizumab, best response partial response
# 1/2021: Palliative radiation to progressive spinal metastases
# 3/2021-present: docetaxel, best response stable disease
Boilerplate:
No evidence of common boilerplate exclusion criteria"""


SUMMARY_FIRST_CHUNK = """\
{system_prompt}

Here is the clinical document for this patient:

<DOCUMENT>
{chunk_text}
</DOCUMENT>

Now, write your clinical summary. Do not add preceding text before the summary, and do not add commentary afterwards."""


SUMMARY_UPDATE_CHUNK = """\
{system_prompt}

You previously wrote the following summary based on earlier portions of this patient's clinical record:

<PRIOR_SUMMARY>
{prior_summary}
</PRIOR_SUMMARY>

Here is the next portion of the patient's clinical record:

<DOCUMENT>
{chunk_text}
</DOCUMENT>

Update the summary to incorporate any new relevant information from this segment.
- If the segment contains no new information, output the prior summary exactly as-is.
- Maintain the same format and structure.
- Do not add preceding text before the summary, and do not add commentary afterwards.

Write the updated summary:"""
