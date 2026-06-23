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
    save_employee_reference_csv,
)
from helper.loader import show_loader, hide_loader
import time


FEATURE_IMPORTANCE_FILENAME = "HAF-EPA_feature_importance.csv"
BALANCED_TRAIN_FILENAME = "HAF-EPA_balanced_training_dataset.csv"
TEST_PREDICTIONS_FILENAME = "HAF-EPA_test_predictions.csv"
RECOMMENDATION_FILENAME = "HAF-EPA_test_recommendations.xlsx"
EVALUATION_FILENAME = "HAF-EPA_model_evaluation.txt"
TOP_EMPLOYEE_FILENAME = "HAF-EPA_top_200_employees.xlsx" 
EMPLOYEE_REFERENCE_FILENAME = "HAF-EPA_employee_reference.csv"


def main():
    print("\n=== STEP 1: preparing labeled dataset ===")
    show_loader()
    labeled_df = prepare_labeled_dataset(performance_threshold=7.0)
    hide_loader()

    print("\n=== STEP 2: Split: Training / Test dataset (80/20) ===")
    train_df, test_df = split_train_test(
        labeled_df=labeled_df,
        feature_columns=FEATURE_COLUMNS,
        test_size=0.20,
        random_state=42,
    )

    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)

    print("\n=== STEP 3: Balance traing dataset  ===")
    balanced_train_df = balance_training_data(
        train_df=train_df,
        negative_multiplier=3,
        random_state=42,
    )
    print("Balanced train shape:", balanced_train_df.shape)
    
    print("\n=== STEP 4: Start train HAF-EPA model ===")
    model, feature_importance_df, model_path = train_haf_epa_model(
        balanced_train_df=balanced_train_df,
        random_state=42,
    )
    print(f"Model saved location: {model_path}")

    print("\n=== STEP 5: Start validate test dataset(20%) ===")
    evaluation_result = evaluate_model(model, test_df)

    print("Accuracy:", evaluation_result["accuracy"])

    #print("\nConfusion Matrix:")
    #print(evaluation_result["confusion_matrix"])
    print("\n== Generate HAF-EPA Model ===")

    print("\nClassification Report:")
    print(evaluation_result["classification_report"])

    print("\n=== STEP 6: Generate hybrid recommendations on test dataset ===")
    recommendation_df = generate_test_recommendations(model, test_df)

    print("\nTop 10 recommendations:")
    print(recommendation_df.head(10))

    print("\n=== STEP 7: Create CSV and Excel files ===")
    show_loader()
    
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
    hide_loader()
    print(f"Feature importance saved to: {feature_importance_path}")
    print(f"Balanced training data saved to: {balanced_train_path}")
    print(f"Test predictions CSV saved to: {test_predictions_path}")
    print(f"Recommendation Excel saved to: {recommendation_excel_path}")
    print(f"Evaluation report saved to: {evaluation_report_path}")

    print("\n=== STEP 8: Create top 200 employee summary Excel sheet ===")

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
    top_employee_csv_path = save_employee_reference_csv(
        top_employee_df=top_employee_df,
        filename=EMPLOYEE_REFERENCE_FILENAME,
    )
    
    
    print(f"Excel saved to: {top_employee_excel_path}")
    print(f"Csv saved to: {top_employee_csv_path}")
    
    print("\n=== TOP 10 iportant features ===")
    print(feature_importance_df.head(10))

    print("\n=== TOP 10 Employees details ===")
    print(top_employee_df.head(10))


    print("\n=== STEP 9: Create top 100 employee ability and availability reference csv file===")
    employee_reference_df = top_employee_df.copy()


    print("\nHAF-EPA PIPELINE COMPLETED SUCCESSFULLY ✅")


if __name__ == "__main__":
    main()