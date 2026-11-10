import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit

from data_processor import (
    DataIngestion,
    DataCleaner,
    FeatureEngineer,
    TARGET_COST_RATIO,
    TARGET_DURATION_RATIO,
    RAW_TARGET_COST,
    RAW_TARGET_DURATION,
)
from model_builder import ConstructionModel, sensitivity_analysis


def main():
    print("Starting Construction Prediction Model Pipeline (Ratio-Based)...")

    # Reproducibility (augmentation noise)
    np.random.seed(42)

    # 1. Ingestion (relative paths, works on Windows / Linux / server)
    BASE = os.path.dirname(os.path.abspath(__file__))
    excel_path = os.path.join(BASE, "Construction_Delay_Template 09-11-2025.xlsx")
    csv_path = os.path.join(BASE, "from web site.csv")

    ingestor = DataIngestion(excel_path, csv_path)
    df_excel, df_csv = ingestor.load_data()

    # 2. Cleaning & Unification
    cleaner = DataCleaner()
    combined_df = cleaner.normalize_columns(df_excel, df_csv)
    combined_df = cleaner.clean(combined_df)

    # Number of ORIGINAL rows (used later to group each project with its noisy copies)
    n = len(combined_df)

    # 3. Augmentation (original + 2 noisy copies)
    print("Applying Augmentation...")
    augmented_list = [combined_df]
    for _ in range(2):
        noisy_df = combined_df.copy()
        numeric_cols = noisy_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in [RAW_TARGET_COST, RAW_TARGET_DURATION]:
                noise = np.random.normal(0, 0.05, size=len(noisy_df))
                noisy_df[col] = noisy_df[col] * (1 + noise)
        augmented_list.append(noisy_df)

    final_df = pd.concat(augmented_list, ignore_index=True)

    # 4. Feature Engineering (calculates ratios here)
    fe = FeatureEngineer()
    processed_df = fe.prepare_features(final_df)

    processed_df = processed_df.replace([np.inf, -np.inf], np.nan)
    processed_df = processed_df.dropna(subset=[TARGET_COST_RATIO, TARGET_DURATION_RATIO])

    print(f"Training Data Shape: {processed_df.shape}")

    # 5. Select Features and Targets
    target_cols = [TARGET_COST_RATIO, TARGET_DURATION_RATIO]
    # Drop targets AND raw targets from input X
    drop_cols = target_cols + [RAW_TARGET_COST, RAW_TARGET_DURATION]
    existing_drop = [c for c in drop_cols if c in processed_df.columns]

    X = processed_df.drop(existing_drop, axis=1)
    y = processed_df[target_cols]

    # 6. Leak-free split:
    # every original project and its noisy copies share the same group id,
    # so they all land together in train OR in test (never both).
    groups = X.index.to_numpy() % n
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

    # Evaluate on ORIGINAL (non-synthetic) projects only
    is_original = X_test.index < n
    X_test, y_test = X_test[is_original], y_test[is_original]

    print(f"Train rows: {len(X_train)} | Test rows (original only): {len(X_test)}")

    # 7. Model Training
    model = ConstructionModel()
    model.build_model()
    model.train(X_train, y_train)

    # 8. Save Model
    save_path = os.path.join(BASE, "construction_model.pkl")
    model.save(save_path)
    print(f"Model saved to {save_path}")
    print(f"Model size: {os.path.getsize(save_path) / 1024 / 1024:.1f} MB")

    # 9. Evaluation (absolute values)
    model.evaluate(X_test, y_test, X_test)

    # 9b. Evaluation on the RATIOS themselves + simple baselines
    pred = model.predict(X_test)
    for i, name in enumerate(target_cols):
        yt = y_test.iloc[:, i].to_numpy()
        base_pred = np.full_like(yt, y_train.iloc[:, i].mean(), dtype=float)
        print(
            f"\n[{name}]  R2 (ratio): {r2_score(yt, pred[:, i]):.4f} | "
            f"MAE (ratio): {mean_absolute_error(yt, pred[:, i]):.4f} | "
            f"baseline MAE (always predict train mean): "
            f"{mean_absolute_error(yt, base_pred):.4f}"
        )

    # Baseline: "no overrun" (ratio = 1) on absolute cost
    exp_cost = X_test["Expected Total Cost"].to_numpy()
    true_cost = y_test.iloc[:, 0].to_numpy() * exp_cost
    print(f"\nBaseline (ratio=1) abs-cost R2: {r2_score(true_cost, exp_cost):.4f}")

    # 10. Sensitivity Analysis
    feature_names = X_train.columns.tolist()
    sensitivity_analysis(model.model, X_test, feature_names)


if __name__ == "__main__":
    main()