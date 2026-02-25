# Earnings Risk Engine — Public Version

## TL;DR — What This Project Is
**Earnings Risk Engine** — a proprietary earnings-risk analytics engine  
Built end-to-end by **Michael Savin** — architecture, engineering & UX.

This Product has been adopted and integrated intro production by a stock portfolio management firm - Tipigo Ventures Ltd.
It analyzes how stocks behave around earnings → quantifies risk, stability & anomalies → produces alerts & recommendations.

I designed and built:

- Designed the architecture & pipeline

- Built the data ingestion & cleaning layers

- Engineered earnings-behaviour & risk features

- Created risk scoring & alert logic (private)

- Output structured dashboard data for investors

- Built a UI for visualization and user experience

Sensitive logic & data integrations are intentionally omitted.
This repo shows structure, design & engineering approach.

# 
# Dashboard Overview
Here’s the main dashboard that sits on top of the analytics engine:

<img width="2557" height="1225" alt="overview" src="https://github.com/user-attachments/assets/12baf5a4-1f2b-45f0-8c30-0ae49dee3615" />

The dashboard is **built in Streamlit**.
# 

# Problem the Product Solves

Earnings events often produce the largest and fastest price moves a stock will make all quarter — but the reactions are noisy and hard to contextualize.
Investors struggle with questions like:
- How does this stock usually react to earnings?
- Is this move normal or excessive?
- Does this stock move with peers or behave independently?
- Is volatility structurally rising or falling?

The platform helps answer those questions through structured risk-aware analytics before and after earnings.


# At a high level, the full product:

- Ingests & cleans historical price, earnings, fundamentals, and sector data
- Builds features around earnings behaviour, volatility, and peer context
- Quantifies risk & stability characteristics per stock and per event
- Generates alerts & recommendations, such as:
  - excessive price moves
  - muted reaction to earnings surprise
  - implied vs. actual reaction divergence
  - peer-level risk clusters
- Outputs an investor-friendly dashboard for scanning risk across names and sectors.
This repo shows the architecture & pipeline design, while omitting the proprietary model logic.

# Dashboard Examples
These screenshots are taken directly from the working dashboard built on top of the analytics engine:

## Overview - Portfolio-level risk scanning & event filtering

<img width="2557" height="1225" alt="overview" src="https://github.com/user-attachments/assets/12baf5a4-1f2b-45f0-8c30-0ae49dee3615" />
 
## Risk Alerts - Automatically flagged anomalies & risk conditions

<img width="2516" height="805" alt="Risk alerts" src="https://github.com/user-attachments/assets/6d6bbd8a-ee75-453e-96e3-23e38c9b874f" />

## Stock Drill-Down - Historical earnings-risk profile for a single stock

<img width="2502" height="1289" alt="Stock_Drill_Down" src="https://github.com/user-attachments/assets/9cde9bb5-e579-4d5d-ab39-728b4b4067f4" />

#

# Architecture Overview
Raw Data

    ↓
    
Stage 1 – Load & Format

    ↓
    
Stage 2 – Earnings Behaviour & Risk Features

    ↓
    
Stage 3 – Risk Assessment & Scoring

    ↓
    
Stage 4 – Recommendations, Alerts & Dashboard Output


# Key Modules (High-Level Only)
- pipeline/
End-to-end orchestration of the analytics workflow.

- pipeline_stage1
Loads & prepares cleaned, earnings-aligned data.

- pipeline_stage2
Builds behaviour- and risk-related features (omitted here).

- pipeline_stage3
Per-earnings & per-stock risk assessment (logic omitted).

- pipeline_stage4
Generates alerts, recommendations & dashboard outputs (logic omitted).

# Tech Stack

- Python
- Pandas / NumPy
- Streamlit (dashboard layer)
- Financial data APIs
- CSV storage

# Data Sources (Described Generically)

The full system works with:
- Historical price data
- Company earnings releases
- Sector / industry classifications
- Company-level risk metrics
- Market indexes

All data sources are standard, reputable financial datasets.
Exact pipelines, API handling, and credentials are not included here.

# Roadmap (Public-Safe Version)

Future private-system work includes:
- Additional structural risk factors
- Regime-aware volatility modelling
- Live data ingestion
- Postgres + cloud storage backend


The full product transforms historical earnings, price reactions, volatility regimes, and sector behaviour into **actionable, risk-aware insights around quarterly earnings events**.

This repository exists to demonstrate:
- software & data engineering architecture
- pipeline design
- analysis approach & domain knowledge
- coding style & structure
without exposing the core proprietary logic.


If you’d like to know more about the system at a conceptual level, I’m happy to discuss — but I cannot share formulas, thresholds, or proprietary logic.

# Contact

If you’d like to discuss:
- data / ML roles
- risk analytics
- fintech products
- startup opportunities

feel free to reach out.

### e-mail: savin992@gmail.com

All rights reserved.
