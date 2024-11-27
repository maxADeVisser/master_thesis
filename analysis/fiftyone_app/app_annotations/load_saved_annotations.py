import ast

import pandas as pd

experiments = ["c30_25D_2411_1543", "c50_25D_2411_1812", "c70_25D_2411_1705"]

for exp in experiments:
    df = pd.read_csv(
        "analysis/fiftyone_app/app_annotations/c30_25D_2411_1543_fold0.csv"
    )
    assert len(df) == 2113
    # df["tags"] = df["tags"].apply(ast.literal_eval)
