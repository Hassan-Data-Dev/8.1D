"""Single source of truth for the SIT720 8.1D Sydney housing dataset schema.

Imported by build_notebook.py, app.py, build_report.py and collect_data.py so
that the collector, the notebook, the deployed Streamlit app and the report can
never disagree about column names, types or allowed values.

DATA PROVENANCE
---------------
Every row is a REAL sold property, collected from domain.com.au sold listings
and realestate.com.au where available. Collection is browser-driven and then
verified row by row against the recorded source_url; see docs/collection_notes.md
for the exact method and its limitations, which are disclosed as assisted
collection.

WHY THERE IS NO LAND SIZE
-------------------------
Domain's listing payload does not expose land size or internal floor area for
sold listings (the field is present in the payload but always zero). That was
verified directly against the page's structured data before this schema was
fixed. Rather than invent the column or leave it mostly empty, the feature set
was pivoted to what is genuinely observable, and this is discussed as a data
limitation in Part 1.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# Paths (all relative to the assessment root, which is the workspace root)
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DOCS_DIR = ROOT / "docs"
FIG_DIR = ROOT / "report_figs"
MODEL_DIR = ROOT / "models"
NOTEBOOK_DIR = ROOT / "notebook"

DATASET_PATH = DATA_DIR / "sydney_houses.csv"
SUBURB_STATS_PATH = DATA_DIR / "suburb_reference.csv"
LLM_PATH = DATA_DIR / "llm_predictions.csv"
HUMAN_PATH = DATA_DIR / "human_estimates.csv"
RAW_PATH = DATA_DIR / "sold_listings_raw.csv"
EXCLUDED_PATH = DATA_DIR / "excluded_listings.csv"
RESULTS_PATH = DATA_DIR / "results.json"

# The app builds its CSV template on the fly from DATASET_PATH, so there is no
# template file on disk. This name is only used as the download filename.
TEMPLATE_DOWNLOAD_NAME = "sydney_houses_template.csv"

NOTEBOOK_NAME = "SIT720_8.1D_226433107-Hassan-Riaz.ipynb"
NOTEBOOK_PATH = NOTEBOOK_DIR / NOTEBOOK_NAME
REPORT_NAME = "SIT720_8.1D_Report_226433107-Hassan-Riaz"

RANDOM_STATE = 42

# --------------------------------------------------------------------------
# Required collection targets (from the brief)
# --------------------------------------------------------------------------
MIN_TOTAL_RECORDS = 100
MIN_PER_SUBURB = 30
TARGET_ROWS = 105  # 35 per suburb: 5% buffer over the 100 minimum

# --------------------------------------------------------------------------
# The three selected suburbs
# Chosen to represent substantially different housing markets (Part 1).
# --------------------------------------------------------------------------
SUBURBS: dict[str, dict[str, object]] = {
    "Mosman": {
        "postcode": 2088,
        "region": "Lower North Shore",
        "distance_to_cbd_km": 8.0,
        "domain_slug": "mosman-nsw-2088",
        "profile": (
            "Premium harbourside market dominated by detached houses on "
            "small-to-medium lots."
        ),
        "why": (
            "Represents the top of the Sydney market: scarce land, harbour and "
            "beach amenity, prestige pricing and low sales volume."
        ),
    },
    "Parramatta": {
        "postcode": 2150,
        "region": "Western Sydney",
        "distance_to_cbd_km": 24.0,
        "domain_slug": "parramatta-nsw-2150",
        "profile": (
            "Sydney's second CBD, high-density apartment market with strong rail "
            "and light-rail access and a large diverse population."
        ),
        "why": (
            "Represents the mid-market investor and owner-occupier segment where "
            "strata apartments dominate and price is driven by internal size, "
            "building amenity and transport access rather than land."
        ),
    },
    "Campbelltown": {
        "postcode": 2560,
        "region": "South-West Sydney",
        "distance_to_cbd_km": 50.0,
        "domain_slug": "campbelltown-nsw-2560",
        "profile": (
            "Affordable outer-metropolitan market of detached family homes with a "
            "young and fast-growing population."
        ),
        "why": (
            "Represents the affordable end of the market where buyers trade "
            "commute time for floor space and land."
        ),
    },
}

# --------------------------------------------------------------------------
# Column schema
# kind: "numeric" | "categorical" | "date" | "text" | "id"
# src : "listing" = available directly from the sold listing
#       "derived" = computed by the collector from the listing
#       "manual"  = filled in by hand during verification
# --------------------------------------------------------------------------
COLUMNS: dict[str, dict[str, object]] = {
    "property_id": {
        "kind": "id", "dtype": "string", "required": True, "src": "derived",
        "desc": "Sequential ID, e.g. MOS-001, PAR-014, CAM-032.",
    },
    "suburb": {
        "kind": "categorical", "dtype": "string", "required": True, "src": "listing",
        "allowed": list(SUBURBS),
        "desc": "Suburb name. Must be one of the three selected suburbs.",
    },
    "postcode": {
        "kind": "categorical", "dtype": "Int64", "required": True, "src": "listing",
        "desc": "Postcode of the suburb.",
    },
    "address": {
        "kind": "text", "dtype": "string", "required": True, "src": "listing",
        "desc": "Street address as listed, e.g. '8/88 Avenue Road'.",
    },
    "sale_price": {
        "kind": "numeric", "dtype": "float64", "required": True, "src": "listing",
        "unit": "AUD",
        "desc": "TARGET. Reported sold price.",
    },
    "sale_date": {
        "kind": "date", "dtype": "datetime64[ns]", "required": True, "src": "listing",
        "desc": "Date the property sold (YYYY-MM-DD).",
    },
    "sale_method": {
        "kind": "categorical", "dtype": "string", "required": False, "src": "listing",
        "allowed": ["privateTreaty", "auction", "priorToAuction",
                    "expressionOfInterest", "unknown"],
        "desc": "How the property sold: private treaty or auction.",
    },
    "property_type": {
        "kind": "categorical", "dtype": "string", "required": True, "src": "listing",
        "allowed": ["House", "Apartment", "Townhouse", "Villa", "Duplex",
                    "SemiDetached", "Terrace"],
        "desc": "Dwelling type, normalised from Domain's propertyType.",
    },
    "bedrooms": {
        "kind": "numeric", "dtype": "Int64", "required": True, "src": "listing",
        "desc": "Number of bedrooms.",
    },
    "bathrooms": {
        "kind": "numeric", "dtype": "Int64", "required": True, "src": "listing",
        "desc": "Number of bathrooms.",
    },
    "car_spaces": {
        "kind": "numeric", "dtype": "Int64", "required": False, "src": "listing",
        "desc": "Number of car spaces. Blank if not advertised.",
    },
    "total_rooms": {
        "kind": "numeric", "dtype": "Int64", "required": False, "src": "derived",
        "desc": "bedrooms + bathrooms, a proxy for dwelling size.",
    },
    "distance_to_cbd_km": {
        "kind": "numeric", "dtype": "float64", "required": True, "src": "derived",
        "desc": "Straight-line distance from the suburb to Sydney GPO.",
    },
    "condition_rating": {
        "kind": "categorical", "dtype": "string", "required": False, "src": "manual",
        "allowed": ["Needs work", "Fair", "Good", "Excellent"],
        "desc": "Condition judged from photos and description during verification.",
    },
    "has_pool": {
        "kind": "numeric", "dtype": "Int64", "required": False, "src": "derived",
        "desc": "1 if the description mentions a pool, else 0.",
    },
    "has_water_view": {
        "kind": "numeric", "dtype": "Int64", "required": False, "src": "derived",
        "desc": "1 if harbour, ocean or river views are claimed, else 0.",
    },
    "agent_description": {
        "kind": "text", "dtype": "string", "required": False, "src": "listing",
        "desc": "Verbatim agent marketing text. Powers the Part 2 text features.",
    },
    "source_url": {
        "kind": "text", "dtype": "string", "required": True, "src": "listing",
        "desc": "Listing URL. MANDATORY for every row: traceability evidence.",
    },
    "verified": {
        "kind": "categorical", "dtype": "string", "required": False, "src": "manual",
        "allowed": ["y", "n"],
        "desc": "Set to 'y' once you have checked the row against source_url.",
    },
}

REQUIRED_COLUMNS = [c for c, m in COLUMNS.items() if m.get("required")]
NUMERIC_COLUMNS = [c for c, m in COLUMNS.items() if m["kind"] == "numeric"]
CATEGORICAL_COLUMNS = [c for c, m in COLUMNS.items() if m["kind"] == "categorical"]
TEXT_COLUMNS = [c for c, m in COLUMNS.items() if m["kind"] == "text"]
ALL_COLUMNS = list(COLUMNS)

TARGET = "sale_price"

# Columns the model may consume. Excludes the target, free identifiers, the raw
# text column (turned into flags separately) and sale_date (used for the Part 2
# trend analysis rather than as a raw predictor).
MODEL_EXCLUDE = {
    "sale_price", "property_id", "source_url", "agent_description",
    "sale_date", "address", "verified",
}

# --------------------------------------------------------------------------
# Engineered features (created in Part 2)
# --------------------------------------------------------------------------
TEXT_KEYWORDS = {
    "kw_renovated": ["renovat", "updated", "modernis", "moderniz", "brand new"],
    "kw_pool": ["pool", "swimming"],
    "kw_water": ["waterfront", "water view", "harbour", "beach", "ocean"],
    "kw_prestige": ["prestige", "luxury", "architect", "designer", "landmark"],
    "kw_quiet": ["quiet", "peaceful", "leafy", "tranquil", "private"],
    "kw_transport": ["transport", "station", "train", "light rail", "bus"],
    "kw_school": ["school", "catchment", "education"],
    "kw_development": ["development", "subdivi", "approved", "potential"],
    "kw_asnew": ["as new", "near new", "immaculate", "pristine"],
}

ENGINEERED_FEATURES = ["total_rooms"] + list(TEXT_KEYWORDS)


def readme() -> str:
    """Human-readable schema summary, used by the collection worksheet."""
    lines = [
        f"{'column':<22} {'req':<4} {'src':<9} {'type':<16} description",
        "-" * 108,
    ]
    for name, meta in COLUMNS.items():
        req = "yes" if meta.get("required") else "no"
        lines.append(
            f"{name:<22} {req:<4} {str(meta.get('src', '')):<9} "
            f"{meta['dtype']:<16} {meta['desc']}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    print(readme())
