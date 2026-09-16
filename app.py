"""SIT720 8.1D Part 6 - Sydney Housing Price Decision Support System.

A Streamlit web application that lets a user enter property features manually or
upload a CSV of properties and receive predicted sale prices from the trained and
persisted ML pipeline.

Run from the assessment root:
    .venv/Scripts/python.exe -m streamlit run app.py

The model is loaded from models/ (written by the notebook in Part 3).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import features
import schema

st.set_page_config(
    page_title="Sydney Housing Price Decision Support System",
    page_icon="🏠",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    """Load the best pipeline produced in Part 3."""
    import joblib

    candidates = sorted(schema.MODEL_DIR.glob("*.joblib"))
    if not candidates:
        return None, None, None
    best = next((p for p in candidates if "best" in p.stem), candidates[-1])
    bundle = joblib.load(best)
    if isinstance(bundle, dict):
        return bundle.get("pipeline"), bundle.get("metadata", {}), best
    return bundle, {}, best


@st.cache_data(show_spinner=False)
def suburb_benchmarks() -> pd.DataFrame:
    """Median price per suburb, computed from the collected dataset."""
    if not schema.DATASET_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(schema.DATASET_PATH)
    if df.empty or "sale_price" not in df.columns:
        return pd.DataFrame()
    return (df.groupby("suburb")["sale_price"]
            .agg(median="median", count="size")
            .round(0))


def main() -> None:
    st.title("🏠 Sydney Housing Price Decision Support System")
    st.caption(
        "SIT720 8.1D Mini Project · Hassan Riaz · Student ID 226433107 · "
        "Deakin University"
    )

    pipeline, meta, model_path = load_model()
    if pipeline is None:
        st.error(
            "**No trained model found in `models/`.**\n\n"
            "Run the Part 3 notebook first (it persists the best pipeline with "
            "`joblib.dump`), then reload this page."
        )
        st.stop()

    with st.sidebar:
        st.header("Model")
        st.write(f"**Artefact:** `{model_path.name}`")
        for key in ("model_name", "cv_rmse", "cv_mae", "cv_r2",
                    "test_rmse", "test_mae", "test_r2", "test_mape"):
            if key in meta:
                value = meta[key]
                label = key.replace("_", " ").upper()
                if isinstance(value, (int, float)):
                    st.write(f"**{label}:** {value:,.3f}")
                else:
                    st.write(f"**{label}:** {value}")
        st.divider()
        st.caption(
            "Estimates are indicative only. This prototype is a university "
            "assessment artefact and is not a formal valuation."
        )

    benchmarks = suburb_benchmarks()
    tab_manual, tab_upload, tab_about = st.tabs(
        ["🔎 Single property", "📄 Batch upload (CSV)", "ℹ️ About & limitations"]
    )

    # ---------------- Single property ----------------
    with tab_manual:
        st.subheader("Enter property details")
        left, mid, right = st.columns(3)

        with left:
            suburb = st.selectbox("Suburb", list(schema.SUBURBS))
            postcode = schema.SUBURBS[suburb]["postcode"]
            property_type = st.selectbox(
                "Property type",
                ["House", "Apartment", "Townhouse", "Villa", "Duplex",
                 "SemiDetached", "Terrace"],
            )
            sale_method = st.selectbox(
                "Sale method",
                ["privateTreaty", "auction", "priorToAuction",
                 "expressionOfInterest", "unknown"],
            )

        with mid:
            bedrooms = st.number_input("Bedrooms", 0, 12, 3)
            bathrooms = st.number_input("Bathrooms", 0, 8, 2)
            car_spaces = st.number_input("Car spaces", 0, 8, 1)
            distance_to_cbd_km = st.number_input(
                "Distance to CBD (km)", 0.0, 80.0,
                float(schema.SUBURBS[suburb]["distance_to_cbd_km"]), 0.5,
            )

        with right:
            condition_rating = st.selectbox(
                "Condition", ["Good", "Excellent", "Fair", "Needs work"])
            has_pool = int(st.checkbox("Swimming pool"))
            has_water_view = int(st.checkbox("Water / harbour view"))

        agent_description = st.text_area(
            "Agent description (optional)",
            placeholder="Paste the listing's marketing text to enable the text "
                        "features...",
            height=90,
        )

        if st.button("Predict sale price", type="primary"):
            values = {
                "suburb": suburb,
                "postcode": postcode,
                "property_type": property_type,
                "sale_method": sale_method,
                "bedrooms": bedrooms,
                "bathrooms": bathrooms,
                "car_spaces": car_spaces,
                "distance_to_cbd_km": distance_to_cbd_km,
                "condition_rating": condition_rating,
                "has_pool": has_pool,
                "has_water_view": has_water_view,
                "agent_description": agent_description,
            }
            try:
                frame = features.build_input_frame(values)
                price = float(pipeline.predict(frame)[0])
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
            else:
                c1, c2 = st.columns(2)
                with c1:
                    st.metric("Predicted sale price", f"${price:,.0f}")
                with c2:
                    if not benchmarks.empty and suburb in benchmarks.index:
                        med = benchmarks.loc[suburb, "median"]
                        st.metric(
                            f"{suburb} median (collected)", f"${med:,.0f}",
                            f"{(price - med) / med * 100:+.1f}% vs prediction",
                        )
                st.caption(
                    "The model cannot see land size, floor area, aspect or "
                    "renovation quality, because those are not published on sold "
                    "listings. See the limitations tab."
                )

    # ---------------- Batch upload ----------------
    with tab_upload:
        st.subheader("Upload a CSV of properties")
        st.write(
            "The file must contain the same columns as the collected dataset. "
            "Only `suburb`, `bedrooms` and `bathrooms` are strictly required."
        )

        if schema.DATASET_PATH.exists():
            st.download_button(
                "⬇️ Download a template from the collected data",
                data=pd.read_csv(schema.DATASET_PATH)
                .head(3).to_csv(index=False).encode("utf-8"),
                file_name=schema.TEMPLATE_DOWNLOAD_NAME,
                mime="text/csv",
            )

        uploaded = st.file_uploader("Choose a CSV file", type=["csv"])
        if uploaded is not None:
            try:
                df = pd.read_csv(uploaded)
            except Exception as exc:
                st.error(f"Could not read the CSV: {exc}")
                return

            st.write(f"Loaded **{len(df)}** rows and **{df.shape[1]}** columns.")
            missing = [c for c in ("suburb", "bedrooms", "bathrooms")
                       if c not in df.columns]
            if missing:
                st.error(f"Missing required column(s): {', '.join(missing)}")
                return

            work = df.copy()
            if "distance_to_cbd_km" not in work.columns:
                work["distance_to_cbd_km"] = work["suburb"].map(
                    {s: m["distance_to_cbd_km"] for s, m in schema.SUBURBS.items()})
            for col in ("property_type", "sale_method", "condition_rating",
                        "agent_description"):
                if col not in work.columns:
                    work[col] = "" if col != "sale_method" else "unknown"
            for col in ("car_spaces", "has_pool", "has_water_view"):
                if col not in work.columns:
                    work[col] = 0

            try:
                enriched = pd.concat(
                    [features.build_input_frame(rec)
                     for rec in work.to_dict(orient="records")],
                    ignore_index=True,
                )
                preds = pipeline.predict(enriched)
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
            else:
                out = df.copy()
                out["predicted_price"] = np.round(preds, 0)
                if schema.TARGET in out.columns:
                    out["error"] = out["predicted_price"] - out[schema.TARGET]
                st.success(f"Predicted prices for {len(out)} properties.")
                st.dataframe(out, width="stretch")
                st.download_button(
                    "⬇️ Download predictions",
                    data=out.to_csv(index=False).encode("utf-8"),
                    file_name="predictions.csv",
                    mime="text/csv",
                )

    # ---------------- About ----------------
    with tab_about:
        st.subheader("How this application was developed")
        st.markdown(
            """
            1. Sold-property data was collected for three Sydney suburbs
               (Mosman, Parramatta, Campbelltown) from domain.com.au.
            2. Features were engineered and three regression models were compared
               using cross-validation.
            3. The best pipeline was serialised with `joblib` into `models/`.
            4. This Streamlit app loads that pipeline and exposes it as a form and
               a batch CSV endpoint.
            """
        )
        st.subheader("Limitations and ethical considerations")
        st.markdown(
            """
            - The dataset is small, so it is **not** representative of the wider
              Sydney market.
            - Roughly nine in ten sold listings withhold their price. Those
              properties are absent, which biases the sample toward sellers who
              published a result.
            - **Land size and floor area are unavailable** from sold listings, so
              the model cannot use the strongest physical driver of house value.
            - Agent descriptions are marketing text and emphasise positives, so
              the text features measure advertised desirability rather than
              objective quality.
            - Distance to the CBD is straight-line and ignores travel time,
              topography and transport access.
            - The model cannot see renovations, aspect, noise or school catchment
              changes.
            - Automated valuations used for lending or pricing can entrench
              historical bias. These predictions are indicative only and must not
              be treated as a formal valuation.
            """
        )


if __name__ == "__main__":
    main()
