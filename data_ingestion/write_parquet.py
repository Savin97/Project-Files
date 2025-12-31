""" Efficiently store your cleaned data for fast use later
This script should:
✔ take cleaned DataFrames
✔ partition by symbol/date if needed
✔ save with stable schema
✔ optionally version datasets
Example layout:

/data/prices/
    year=2023/
        prices_2023.parquet
/data/earnings/
    earnings.parquet
"""