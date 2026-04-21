from __future__ import annotations

from data_loader.load_datasets import load_datasets
from process.normalize import normalize_loaded_data

from models.train_model import train_haf_epa_model, FEATURE_COLUMNS
from pipeline.prepare_dataset import prepare_labeled_dataset
from pipeline.split_data import split_train_test
from pipeline.balance_data import balance_training_data
from pipeline.evaluate import evaluate_model
from pipeline.recommend import generate_test_recommendations
from pipeline.export_results import (
    save_feature_importance,
    save_balanced_training_data,
    save_test_predictions_csv,
    save_recommendation_excel,
    save_evaluation_report,
)
from pipeline.export_top_employees import (
    create_top_employee_summary,
    save_top_employee_summary_excel,
)


FEATURE_IMPORTANCE_FILENAME = "HAF-EPA_feature_importance.csv"
BALANCED_TRAIN_FILENAME = "HAF-EPA_balanced_training_dataset.csv"
TEST_PREDICTIONS_FILENAME = "HAF-EPA_test_predictions.csv"
RECOMMENDATION_FILENAME = "HAF-EPA_test_recommendations.xlsx"
EVALUATION_FILENAME = "HAF-EPA_model_evaluation.txt"
TOP_EMPLOYEE_FILENAME = "HAF-EPA_top_200_employees.xlsx"


def main():
    print("\n=== STEP 1: PREPARE LABELED DATASET ===")
    labeled_df = prepare_labeled_dataset(performance_threshold=7.0)
    print("Labeled dataset shape:", labeled_df.shape)

    print("\nFull label distribution:")
    print(labeled_df["label"].value_counts())

    print("\n=== STEP 2: TRAIN / TEST SPLIT (80/20, NO LEAKAGE) ===")
    train_df, test_df = split_train_test(
        labeled_df=labeled_df,
        feature_columns=FEATURE_COLUMNS,
        test_size=0.20,
        random_state=42,
    )

    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)

    print("\nTrain label distribution (before balancing):")
    print(train_df["label"].value_counts())

    print("\nTest label distribution:")
    print(test_df["label"].value_counts())

    print("\n=== STEP 3: BALANCE TRAINING SET ONLY ===")
    balanced_train_df = balance_training_data(
        train_df=train_df,
        negative_multiplier=3,
        random_state=42,
    )

    print("Balanced train shape:", balanced_train_df.shape)

    print("\nBalanced train label distribution:")
    print(balanced_train_df["label"].value_counts())

    print("\n=== STEP 4: TRAIN HAF-EPA MODEL ===")
    model, feature_importance_df, model_path = train_haf_epa_model(
        balanced_train_df=balanced_train_df,
        random_state=42,
    )
    print(f"Model saved to: {model_path}")

    print("\n=== STEP 5: TEST ON HELD-OUT 20% DATA ===")
    evaluation_result = evaluate_model(model, test_df)

    print("Accuracy:", evaluation_result["accuracy"])

    print("\nConfusion Matrix:")
    print(evaluation_result["confusion_matrix"])

    print("\nClassification Report:")
    print(evaluation_result["classification_report"])

    print("\n=== STEP 6: GENERATE HYBRID RECOMMENDATIONS ON TEST SET ===")
    recommendation_df = generate_test_recommendations(model, test_df)

    print("\nTop 10 recommendations:")
    print(recommendation_df.head(10))

    print("\n=== STEP 7: SAVE OUTPUTS ===")
    feature_importance_path = save_feature_importance(
        feature_importance_df,
        FEATURE_IMPORTANCE_FILENAME,
    )

    balanced_train_path = save_balanced_training_data(
        balanced_train_df,
        BALANCED_TRAIN_FILENAME,
    )

    test_predictions_path = save_test_predictions_csv(
        recommendation_df,
        TEST_PREDICTIONS_FILENAME,
    )

    recommendation_excel_path = save_recommendation_excel(
        recommendation_df,
        RECOMMENDATION_FILENAME,
    )

    evaluation_report_path = save_evaluation_report(
        filename=EVALUATION_FILENAME,
        accuracy=evaluation_result["accuracy"],
        confusion_mat=evaluation_result["confusion_matrix"],
        class_report=evaluation_result["classification_report"],
        train_shape=train_df.shape,
        test_shape=test_df.shape,
        balanced_train_shape=balanced_train_df.shape,
        train_label_counts=train_df["label"].value_counts(),
        test_label_counts=test_df["label"].value_counts(),
    )

    print(f"Feature importance saved to: {feature_importance_path}")
    print(f"Balanced training data saved to: {balanced_train_path}")
    print(f"Test predictions CSV saved to: {test_predictions_path}")
    print(f"Recommendation Excel saved to: {recommendation_excel_path}")
    print(f"Evaluation report saved to: {evaluation_report_path}")

    print("\n=== STEP 8: CREATE TOP 200 EMPLOYEE SUMMARY EXCEL ===")

    # Reload source data for employee-level export
    raw_data = load_datasets()
    raw_data = normalize_loaded_data(raw_data)

    top_employee_df = create_top_employee_summary(
        recommendation_df=recommendation_df,
        employees_df=raw_data.employees,
        employee_skills_df=raw_data.employee_skills,
        skills_df=raw_data.skills,
        employee_project_history_df=raw_data.employee_project_history,
        employee_availability_df=raw_data.employee_availability,
        employee_relationship_df=raw_data.employee_relationship,
        top_n=200,
    )

    top_employee_excel_path = save_top_employee_summary_excel(
        top_employee_df=top_employee_df,
        filename=TOP_EMPLOYEE_FILENAME,
    )

    print(f"Top 200 employee Excel saved to: {top_employee_excel_path}")

    print("\n=== TOP 10 FEATURE IMPORTANCE ===")
    print(feature_importance_df.head(10))

    print("\n=== TOP 10 EMPLOYEES ===")
    print(top_employee_df.head(10))

    print("\nHAF-EPA PIPELINE COMPLETED SUCCESSFULLY ✅")


if __name__ == "__main__":
    main()