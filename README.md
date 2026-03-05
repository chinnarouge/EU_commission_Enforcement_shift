# EU Competition Case Study — Analysis of European Commission Press Releases

## Overview

This project analyses approximately 25 years of European Commission (EC) competition enforcement press releases. Raw HTML press releases are parsed, structured, classified, and then subjected to statistical analysis, machine learning, and network analysis to uncover trends, anomalies, and predictive patterns in EU competition policy.

## Project Structure

```
├── data_preprocessing/          # Data ingestion and preparation pipeline
│   ├── pipeline.bat             # Runs all preprocessing steps in order
│   ├── data_extraction.py       # Extracts .7z archives into raw HTML
│   ├── data_preprocessing.py    # Parses HTML into structured CSV (NER, regex, etc.)
│   ├── classify.py              # Classifies cases as merger/antitrust/state aid
│   └── data_overview_and _postprocessing.py
│                                # Assigns sector, decision stage, outcome; saves final CSV
│
├── ml_models/                   # Analysis and modelling scripts
│   ├── complete_analysis.py     # Full statistical overview, correlation, plots
│   ├── outcome_predictor.py     # Predicts investigation outcomes (Logistic Regression)
│   ├── anomoly_detection.py     # Isolation Forest anomaly detection
│   ├── network_analysis.py      # Citation network (NetworkX, directed graph)
│   ├── topic_modeling.py        # LDA topic modelling across eras
│   └── relationship_finder.py   # Apriori association rule mining
│
├── data/
│   ├── Original_zip_files/      # Source .7z archives (not tracked)
│   ├── raw_data/                # Extracted HTML press releases
│   ├── processesd_data/         # Intermediate and final CSVs
│   └── plots/                   # All generated charts and figures
│
├── requirements.txt
└── README.md
```

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Run the preprocessing pipeline

Place the `.7z` archives in `data/Original_zip_files/`, then:

```bash
data_preprocessing\pipeline.bat
```

This runs four steps in order:
1. Extract `.7z` archives to `data/raw_data/`
2. Parse HTML files into `data/processesd_data/structured_data.csv`
3. Classify cases and filter competition cases
4. Assign sectors, decision stages, and binary outcomes, producing `data/processesd_data/final_competition_cases.csv`

### 3. Run analysis scripts

After preprocessing, run any model script independently:

```bash
python ml_models/complete_analysis.py
python ml_models/outcome_predictor.py
python ml_models/anomoly_detection.py
python ml_models/network_analysis.py
python ml_models/topic_modeling.py
python ml_models/relationship_finder.py
```

All plots are saved to `data/plots/`.

## Key Methods

- Text extraction and NER using spaCy and BeautifulSoup
- TF-IDF + Logistic Regression for case classification and outcome prediction
- Keyword-based sector and decision-stage assignment
- Isolation Forest for anomaly detection
- LDA for topic modelling
- Apriori algorithm for association rule mining
- NetworkX directed graph for case citation network analysis
- Cramer's V, HHI, z-score for statistical analysis

## Requirements

- Python 3.9 or later
- See requirements.txt for package list
- spaCy model: en_core_web_sm
