# Pipeline module to orchestrate the data processing stages

from pipeline.pipeline_stage1 import load_and_format_data, add_stage_1_features
from pipeline.pipeline_stage2 import stage2
from pipeline.pipeline_stage3 import stage3
from pipeline.pipeline_stage4 import stage4

def run_pipeline():
    data = load_and_format_data()
    data_stage_1 = add_stage_1_features(data)
    engineered  = stage2(data_stage_1)
    earnings_df = stage3(engineered )
    dashboard_df = stage4(earnings_df)
    return dashboard_df