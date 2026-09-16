# SIT720 8.1D - Sydney Housing Price Prediction and Decision Support System

Hassan Riaz | Student ID 226433107 | Deakin University, SIT720 Machine Learning, T2 2026

A complete machine learning mini project: manual data collection across three
contrasting Sydney suburbs, feature engineering, three regression models compared
under cross-validation, prediction-failure analysis, a three-way benchmark of
machine learning against a large language model and human judgement, and a
deployed Streamlit application.

## What is in this folder

| Path | Purpose |
|---|---|
|
| `schema.py` | **Single source of truth** for the dataset schema, suburbs and constants |
| `features.py` | **Shared** data loading and feature engineering used by the notebook and the app |
| `app.py` | Streamlit application (Part 6) |
| `data/` | Collected dataset, suburb reference, Part 5 recording sheets |
| `models/` | Serialised model written by the notebook |
| `notebook/` | The  notebook |
## Setup

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

Python 3.12 is assumed.

