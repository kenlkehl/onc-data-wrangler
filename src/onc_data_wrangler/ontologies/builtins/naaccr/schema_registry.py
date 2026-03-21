"""Schema Registry: maps cancer site/histology to required site-specific data items.

Ported from onc-registry-extraction/naaccr_pipeline/dictionary/schema_registry.py.
Extended with additional schemas from the existing onc_data_wrangler NAACCR ontology.
Implements the ``SchemaResolverLike`` protocol.
"""

from __future__ import annotations

from typing import Optional
import logging
import re

logger = logging.getLogger(__name__)

_SITE_PREFIX_RE = re.compile(r"^(C\d{2})", re.IGNORECASE)


class SchemaRegistry:
    """Maps primary site + histology -> cancer schema -> required SSDIs.

    Implements the ``SchemaResolverLike`` protocol via ``resolve_schema()``,
    ``get_schema_items()``, and ``get_schema_context()``.
    """

    # ------------------------------------------------------------------
    # Core staging items that apply to ALL cancer types
    # ------------------------------------------------------------------

    CORE_STAGING_ITEMS: list[int] = [
        752, 754, 756, 764, 772, 774, 776, 820, 830, 832, 834, 835,
        880, 890, 900, 910, 940, 950, 960, 970,
        1001, 1002, 1003, 1004, 1011, 1012, 1013, 1014, 1060,
        1112, 1113, 1114, 1115, 1116, 1117, 1182,
        3843, 3844,
    ]

    # ------------------------------------------------------------------
    # Site-specific data items (SSDIs) by schema
    # ------------------------------------------------------------------

    SCHEMA_SSDI_MAP: dict[str, list[int]] = {
        "breast": [
            3827, 3826, 3828, 3915, 3914, 3916,
            3850, 3851, 3852, 3853, 3854, 3855,
            3863, 3894, 3895, 3903, 3904, 3905, 3906,
            3882, 3922,
        ],
        "prostate": [
            3838, 3839, 3840, 3841, 3842, 3920, 3897, 3898, 3919,
        ],
        "colon_rectum": [
            3823, 3819, 3820, 3909, 3890, 3866, 3934, 3929,
        ],
        "lung": [
            3938, 3939, 3866, 3940, 1174, 1176, 3937, 3929,
        ],
        "melanoma_skin": [3817, 3893, 3936],
        "kidney_renal_pelvis": [3925],
        "bladder": [3922],
        "thyroid": [3830, 3833],
        "cervix": [3836, 3956],
        "ovary": [3818, 3836, 3921, 3911],
        "testis": [
            3807, 3808, 3805, 3806, 3848, 3849, 3846, 3847,
            3868, 3867, 3923, 3924,
        ],
        "liver": [3809, 3810, 3835],
        "pancreas": [3942],
        "head_neck": [3831, 3832, 3956],
        "brain_cns": [3816],
        # Extended schemas from onc_data_wrangler (not in registry)
        "esophagus": [3830, 3833, 3922],
        "stomach": [3830, 3833, 3890],
        "uterus": [3836],
        "lymphoma": [],
        "leukemia": [],
        "myeloma": [],
    }

    # ------------------------------------------------------------------
    # Site -> schema map
    # ------------------------------------------------------------------

    SITE_SCHEMA_MAP: dict[str, str] = {
        "C50": "breast",
        "C61": "prostate",
        "C18": "colon_rectum", "C19": "colon_rectum",
        "C20": "colon_rectum", "C21": "colon_rectum",
        "C34": "lung",
        "C44": "melanoma_skin",
        "C64": "kidney_renal_pelvis", "C65": "kidney_renal_pelvis",
        "C67": "bladder",
        "C73": "thyroid",
        "C53": "cervix",
        "C56": "ovary",
        "C62": "testis",
        "C22": "liver",
        "C25": "pancreas",
        # Head and neck
        "C00": "head_neck", "C01": "head_neck", "C02": "head_neck",
        "C03": "head_neck", "C04": "head_neck", "C05": "head_neck",
        "C06": "head_neck", "C07": "head_neck", "C08": "head_neck",
        "C09": "head_neck", "C10": "head_neck", "C11": "head_neck",
        "C12": "head_neck", "C13": "head_neck", "C14": "head_neck",
        "C30": "head_neck", "C31": "head_neck", "C32": "head_neck",
        # Brain/CNS
        "C70": "brain_cns", "C71": "brain_cns", "C72": "brain_cns",
        # Extended sites
        "C15": "esophagus",
        "C16": "stomach",
        "C54": "uterus", "C55": "uterus",
    }

    _MELANOMA_HIST_LO = 8720
    _MELANOMA_HIST_HI = 8790

    # Hematologic malignancy histology ranges
    _LYMPHOMA_HIST = (9590, 9729)
    _LEUKEMIA_HIST = (9731, 9948)  # overlaps myeloma
    _MYELOMA_HIST = (9731, 9734)

    # ------------------------------------------------------------------
    # Site-specific extraction context for LLM prompts
    # ------------------------------------------------------------------

    _SITE_CONTEXT: dict[str, str] = {
        "breast": (
            "For breast cancer, extract:\n"
            "- ER (Estrogen Receptor) status: summary (positive/negative/borderline), "
            "percent positive (exact value if stated, e.g. 95%%), Allred score (0-8).\n"
            "- PR (Progesterone Receptor) status: summary, percent positive, Allred score.\n"
            "- HER2 status: IHC score (0, 1+, 2+, 3+), ISH result (positive/negative, "
            "dual probe copy number and ratio, single probe copy number), overall summary.\n"
            "- Ki-67 proliferation index (percentage if stated).\n"
            "- Multigene signature: method (Oncotype Dx, MammaPrint, Prosigna/PAM50, "
            "EndoPredict, Breast Cancer Index), results/recurrence score, and risk category.\n"
            "- Oncotype Dx recurrence score (0-100) for both DCIS and invasive, with risk level.\n"
            "- Axillary lymph node involvement: number of positive nodes at Level I-II.\n"
            "- Response to neoadjuvant therapy if applicable.\n"
            "Receptor percentages should be exact values when available."
        ),
        "prostate": (
            "For prostate cancer, extract:\n"
            "- Gleason patterns: both clinical and pathological. Record the two pattern numbers.\n"
            "- Gleason score: clinical and pathological (sum of patterns).\n"
            "- Gleason tertiary pattern if mentioned.\n"
            "- PSA lab value: most recent pre-treatment PSA in ng/mL.\n"
            "- Number of biopsy cores examined and number positive.\n"
            "- EOD prostate pathologic extension.\n"
            "Distinguish between clinical Gleason (biopsy) and pathological Gleason (prostatectomy)."
        ),
        "colon_rectum": (
            "For colorectal cancer, extract:\n"
            "- CEA: pretreatment lab value (ng/mL) and interpretation.\n"
            "- Circumferential resection margin (CRM): distance in mm.\n"
            "- Microsatellite instability (MSI): MSI-H/MSI-L/MSS.\n"
            "- KRAS mutation status.\n"
            "- Perineural invasion.\n"
            "- Tumor deposits: number of discrete tumor deposits.\n"
            "- Separate tumor nodules."
        ),
        "lung": (
            "For lung cancer, extract:\n"
            "- ALK rearrangement: positive/negative.\n"
            "- EGFR mutations: specific mutations if stated.\n"
            "- KRAS mutation: specific mutation (e.g. G12C).\n"
            "- BRAF mutation: V600E or other.\n"
            "- PD-L1 expression: TPS percentage.\n"
            "- Spread through air spaces (STAS).\n"
            "- Visceral pleural invasion.\n"
            "- Separate tumor nodules.\n"
            "Record specific mutations, not just positive/negative."
        ),
        "melanoma_skin": (
            "For cutaneous melanoma, extract:\n"
            "- Breslow tumor thickness: depth in mm.\n"
            "- Mitotic rate: number per mm2.\n"
            "- Ulceration: present or absent."
        ),
        "kidney_renal_pelvis": (
            "For kidney/renal pelvis cancer, extract:\n"
            "- Sarcomatoid features: percentage or present/absent."
        ),
        "bladder": (
            "For bladder cancer, extract:\n"
            "- Response to neoadjuvant therapy if given.\n"
            "Note whether muscle-invasive (T2+) vs non-muscle-invasive."
        ),
        "thyroid": (
            "For thyroid cancer, extract:\n"
            "- Extranodal extension: clinical and pathological."
        ),
        "cervix": (
            "For cervical cancer, extract:\n"
            "- FIGO stage.\n"
            "- p16 immunohistochemistry."
        ),
        "ovary": (
            "For ovarian cancer, extract:\n"
            "- CA-125 pretreatment interpretation and lab value.\n"
            "- FIGO stage.\n"
            "- Residual tumor volume post cytoreduction.\n"
            "- Peritoneal cytology."
        ),
        "testis": (
            "For testicular cancer, extract:\n"
            "- AFP: pre- and post-orchiectomy lab values and ranges.\n"
            "- hCG: pre- and post-orchiectomy lab values and ranges.\n"
            "- LDH: pre- and post-orchiectomy ranges.\n"
            "- S Category: clinical and pathological."
        ),
        "liver": (
            "For liver cancer, extract:\n"
            "- AFP pretreatment interpretation and lab value.\n"
            "- Fibrosis score (Ishak score)."
        ),
        "pancreas": (
            "For pancreatic cancer, extract:\n"
            "- CA 19-9 pretreatment lab value in U/mL."
        ),
        "head_neck": (
            "For head and neck cancer, extract:\n"
            "- Extranodal extension: clinical and pathological.\n"
            "- p16 status: important for oropharyngeal cancers."
        ),
        "brain_cns": (
            "For brain/CNS tumors, extract:\n"
            "- Brain molecular markers: IDH1/IDH2, 1p/19q codeletion, "
            "MGMT methylation, ATRX loss, H3K27M."
        ),
        "esophagus": (
            "For esophageal cancer, extract:\n"
            "- Extranodal extension.\n"
            "- Response to neoadjuvant therapy.\n"
            "Note Barrett's esophagus status and histologic subtype."
        ),
        "stomach": (
            "For gastric cancer, extract:\n"
            "- Extranodal extension.\n"
            "- Microsatellite instability.\n"
            "Note HER2 status for advanced gastric cancer."
        ),
        "uterus": (
            "For uterine/endometrial cancer, extract:\n"
            "- FIGO stage.\n"
            "Note histologic type and grade."
        ),
    }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_schema_for_site_histology(
        self,
        primary_site: str,
        histology: str,
        schema_discriminator: Optional[str] = None,
    ) -> str:
        """Determine schema from ICD-O-3 topography + morphology."""
        prefix = self._normalize_site_prefix(primary_site)
        if not prefix:
            # Check for hematologic malignancies by histology
            return self._check_heme_histology(histology)

        schema = self.SITE_SCHEMA_MAP.get(prefix)
        if schema is None:
            return self._check_heme_histology(histology)

        # Special case: skin (C44) requires melanoma histology
        if prefix == "C44":
            hist_num = self._parse_histology(histology)
            if hist_num is None or not (
                self._MELANOMA_HIST_LO <= hist_num <= self._MELANOMA_HIST_HI
            ):
                return "generic"

        return schema

    def _check_heme_histology(self, histology: str) -> str:
        """Check if histology indicates a hematologic malignancy."""
        hist_num = self._parse_histology(histology)
        if hist_num is None:
            return "generic"
        if self._MYELOMA_HIST[0] <= hist_num <= self._MYELOMA_HIST[1]:
            return "myeloma"
        if self._LYMPHOMA_HIST[0] <= hist_num <= self._LYMPHOMA_HIST[1]:
            return "lymphoma"
        if 9800 <= hist_num <= 9948:
            return "leukemia"
        return "generic"

    def get_required_ssdis(self, schema: str) -> list[int]:
        return list(self.SCHEMA_SSDI_MAP.get(schema, []))

    def get_all_staging_items(self, schema: str) -> list[int]:
        """Return core staging items + schema-specific SSDIs, deduplicated."""
        core = list(self.CORE_STAGING_ITEMS)
        ssdis = self.get_required_ssdis(schema)
        seen: set[int] = set(core)
        combined = list(core)
        for item_num in ssdis:
            if item_num not in seen:
                seen.add(item_num)
                combined.append(item_num)
        return combined

    def get_site_context(self, schema: str) -> str:
        context = self._SITE_CONTEXT.get(schema)
        if context:
            return context
        return (
            "Extract all available staging information including TNM stage, "
            "Summary Stage 2018, EOD fields, tumor size, regional lymph node "
            "status, and any biomarkers or prognostic factors mentioned."
        )

    def get_primary_site_description(self, schema: str) -> str:
        descriptions: dict[str, str] = {
            "breast": "breast",
            "prostate": "prostate",
            "colon_rectum": "colorectal",
            "lung": "lung/bronchus",
            "melanoma_skin": "cutaneous melanoma",
            "kidney_renal_pelvis": "kidney/renal pelvis",
            "bladder": "urinary bladder",
            "thyroid": "thyroid",
            "cervix": "cervix uteri",
            "ovary": "ovary/fallopian tube",
            "testis": "testis",
            "liver": "liver/intrahepatic bile duct",
            "pancreas": "pancreas",
            "head_neck": "head and neck",
            "brain_cns": "brain/central nervous system",
            "esophagus": "esophagus",
            "stomach": "stomach",
            "uterus": "uterus/endometrium",
            "lymphoma": "lymphoma",
            "leukemia": "leukemia",
            "myeloma": "multiple myeloma",
            "generic": "cancer (site not specified)",
        }
        return descriptions.get(schema, "cancer")

    # ------------------------------------------------------------------
    # SchemaResolverLike protocol
    # ------------------------------------------------------------------

    def resolve_schema(self, context: dict[str, str]) -> str:
        """SchemaResolverLike protocol: resolve schema from context dict."""
        return self.get_schema_for_site_histology(
            context.get("primary_site", ""),
            context.get("histology", ""),
            context.get("schema_discriminator"),
        )

    def get_schema_items(self, schema: str) -> list[str]:
        """SchemaResolverLike protocol: return field_ids for schema."""
        return [str(n) for n in self.get_all_staging_items(schema)]

    def get_schema_context(self, schema: str) -> str:
        """SchemaResolverLike protocol: alias for get_site_context."""
        return self.get_site_context(schema)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_site_prefix(primary_site: str) -> Optional[str]:
        if not primary_site:
            return None
        cleaned = primary_site.strip().replace(" ", "")
        m = _SITE_PREFIX_RE.match(cleaned)
        if m:
            return m.group(1).upper()
        return None

    @staticmethod
    def _parse_histology(histology: str) -> Optional[int]:
        if not histology:
            return None
        cleaned = histology.strip().split("/")[0].strip()
        try:
            return int(cleaned)
        except (ValueError, TypeError):
            return None
