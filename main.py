# main.py

from pipeline.pipeline import run_pipeline
from config import DASHBOARD_OUTPUT_FILE_PATH

def main():
  dashboard_df = run_pipeline()
  dashboard_df.to_csv(DASHBOARD_OUTPUT_FILE_PATH, index=False)
  print("\nDone.\n\n\n")

if __name__ == "__main__":
  main()