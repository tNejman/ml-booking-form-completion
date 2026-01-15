import json
from typing import List

import numpy as np
import pandas as pd

PREDICTION_LOG_FILE = "ab_test_logs.jsonl"
FEEDBACK_LOG_FILE = "feedback_logs.jsonl"

CATEGORICAL_VARS = ["room_type", "property_type", "bathrooms_text"]
NUMERICAL_VARS = ["bedrooms", "beds", "accommodates"]
LIST_VARS = ["amenities"]


def load_jsonl(file_path: str) -> List[dict]:
    data = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return []
    return data


def calculate_jaccard(list1: List, list2: List) -> float:
    s1 = set(list1) if isinstance(list1, list) else set()
    s2 = set(list2) if isinstance(list2, list) else set()

    if not s1 and not s2:
        return 1.0

    intersection = len(s1.intersection(s2))
    union = len(s1.union(s2))
    return intersection / union if union > 0 else 0.0


def main():
    print("--- Loading Logs ---")
    pred_logs = load_jsonl(PREDICTION_LOG_FILE)
    feedback_logs = load_jsonl(FEEDBACK_LOG_FILE)

    if not pred_logs or not feedback_logs:
        print("Insufficient data to run analysis.")
        return

    df_preds = pd.DataFrame(pred_logs)
    df_feedbacks = pd.DataFrame(feedback_logs)

    preds_normalized = pd.json_normalize(df_preds["prediction"])
    preds_normalized.columns = [f"pred_{col}" for col in preds_normalized.columns]

    df_preds_flat = pd.concat(
        [df_preds[["prediction_id", "model_used"]], preds_normalized], axis=1
    )

    rename_map = {
        col: f"actual_{col}" for col in df_feedbacks.columns if col != "prediction_id"
    }
    df_feedbacks_renamed = df_feedbacks.rename(columns=rename_map)

    merged_df = pd.merge(
        df_preds_flat, df_feedbacks_renamed, on="prediction_id", how="inner"
    )

    print(f"Total merged records (Prediction + Feedback): {len(merged_df)}")

    if len(merged_df) == 0:
        print("No matching prediction IDs found between prediction and feedback logs.")
        return

    models = merged_df["model_used"].unique()

    results_table = []

    for model in models:
        model_df = merged_df[merged_df["model_used"] == model]
        stats = {"Model": model, "Count": len(model_df)}

        for var in CATEGORICAL_VARS:
            pred_col = f"pred_{var}"
            actual_col = f"actual_{var}"

            if pred_col in model_df.columns and actual_col in model_df.columns:
                y_pred = model_df[pred_col].fillna("MISSING").astype(str)
                y_true = model_df[actual_col].fillna("MISSING").astype(str)

                accuracy = (y_pred == y_true).mean()
                stats[f"Acc_{var}"] = round(accuracy, 4)
            else:
                stats[f"Acc_{var}"] = None

        for var in NUMERICAL_VARS:
            pred_col = f"pred_{var}"
            actual_col = f"actual_{var}"

            if pred_col in model_df.columns and actual_col in model_df.columns:
                valid_rows = model_df[[pred_col, actual_col]].dropna()

                if not valid_rows.empty:
                    mae = np.mean(np.abs(valid_rows[pred_col] - valid_rows[actual_col]))
                    stats[f"MAE_{var}"] = round(mae, 4)
                else:
                    stats[f"MAE_{var}"] = None
            else:
                stats[f"MAE_{var}"] = None

        if (
            "pred_amenities" in model_df.columns
            and "actual_amenities" in model_df.columns
        ):
            jaccard_scores = model_df.apply(
                lambda row: calculate_jaccard(
                    row["pred_amenities"], row["actual_amenities"]
                ),
                axis=1,
            )
            stats["Jaccard_amenities"] = round(jaccard_scores.mean(), 4)
        else:
            stats["Jaccard_amenities"] = None

        results_table.append(stats)

    results_df = pd.DataFrame(results_table)

    results_df = results_df.set_index("Model").T

    print("\n" + "=" * 50)
    print(" A/B TEST RESULTS SUMMARY ")
    print("=" * 50)
    print(results_df)
    print("=" * 50)


if __name__ == "__main__":
    main()
