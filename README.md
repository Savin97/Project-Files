Earnings-Based Risk & Volatility Tracker — AI-Assisted Decision-Support System

Markets misprice earnings risk because most tools only look at current earnings — not historical reaction patterns, sector-context behaviour, surprise sensitivity, and tail-risk frequency.
This tool quantifies this hidden risk and flags abnormal reactions before and after earnings.

Summary
This project is an end-to-end Python workflow that analyzes stock price reactions around earnings events to surface risk, volatility patterns, and anomalous market behavior for decision-makers.
It is designed as a modular, production-oriented analytics system with reusable ingestion, feature engineering, scoring, and alerting logic.

The system ingests historical earnings and market data, engineers multi-window reaction features (returns, volatility, drift, momentum), and generates structured event-level risk signals and alerts that can be integrated into portfolio workflows.

### Quickstart

1. Create a virtual env and install dependencies:
    ```bash
    pip install -r requirements.txt
    Put your CSVs in data/ (see config.py for filenames).
    Run the pipeline:
    python main.py
    This will produce outputs/output_dashboard_ready.csv, which you can connect to Tableau.
    (Optional) Launch the Streamlit dashboard:
    streamlit run streamlit/app.py
