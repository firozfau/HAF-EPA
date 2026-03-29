from __future__ import annotations

from models.generate_train_model import generate_train_model
from models.evaluate_model import evaluate_model
from models.test_model import test_model
from knowledge_graph.kg_recommend import kg_recommend
from models.final_recommendation import generate_final_recommendation
from models.hybrid_recommendation import hybrid_recommendation
from helper.model_required import is_traing_model_available
from config import (
    FINAL_RECOMMENDED_EXCEL,
    HYBRID_RECOMMENDED_EXCEL,
    KNOWLEDGE_RECOMMENDED_EXCEL,
)


# =========================================================
# PIPELINE CONTROL SWITCHES
# =========================================================

# 1. Training switch
# Turn ON only when you want to retrain the model.
RUN_GENERATE_TRAIN_MODEL = True

# 2. Internal held-out evaluation (20% test split)
RUN_EVALUATE_MODEL = True

# 3. External test dataset prediction + external metrics
RUN_TEST_MODEL = True

# 4. Knowledge Graph recommendation
RUN_KG_RECOMMEND = True

# 5. Final recommendation (ML only)
RUN_FINAL_RECOMMENDATION = True

# 6. Hybrid recommendation (ML + KG)
RUN_HYBRID_RECOMMENDATION = True


# =========================================================
# PROJECT SELECTION AND OUTPUT CONTROL
# =========================================================

selected_project_id = "P00024"
top_k = 10
total_number_hybrid = 10


def main() -> None:
    print("\nHAF-EPA Main Controller Starting...\n")

    graph_rec = None
    predicted_DataFrame = None
    final_rec = None
    hybrid_rec = None

    # =====================================================
    # 1. Training stage
    # =====================================================
    if RUN_GENERATE_TRAIN_MODEL:
        print("1) Training stage started...")
        trained = generate_train_model()

        print("   => Training completed successfully")
        print("   => Train size:", len(trained.X_train))
        print("   => Held-out internal test size:", len(trained.X_test))
        print("   => Training threshold:", trained.threshold)


    if not is_traing_model_available():
        print(" ==> No trained model found. Please train the model before proceeding.")
    else:

        # =====================================================
        # 2. Internal evaluation stage
        # =====================================================
        if RUN_EVALUATE_MODEL:
            print("\n2) Internal evaluation stage started...")
            evaluate_metrics = evaluate_model()

            print("   => Internal evaluation completed successfully")
            print("   => Model Accuracy :", evaluate_metrics.accuracy)
            print("   => Precision      :", evaluate_metrics.precision)
            print("   => Recall         :", evaluate_metrics.recall)
            print("   => F1-score       :", evaluate_metrics.f1)
            print("   => Threshold      :", evaluate_metrics.threshold)
            print("   => Confusion Matrix:", evaluate_metrics.confusion_matrix)
            print("\n   => Classification Report:")
            print(evaluate_metrics.classification_report)
        else:
            print("\n2) Internal evaluation stage skipped.")

        # =====================================================
        # 3. Knowledge Graph recommendation
        # =====================================================
        if RUN_KG_RECOMMEND:
            print("\n3) Knowledge Graph recommendation stage started...")
            graph_rec = kg_recommend()

            if graph_rec is not None and not graph_rec.empty:
                graph_rec.to_excel(KNOWLEDGE_RECOMMENDED_EXCEL, index=False)
                print("   => Knowledge Graph recommendation completed successfully")
                print(f"   => Saved: {KNOWLEDGE_RECOMMENDED_EXCEL}")
            else:
                print("   => KG recommendation returned empty data.")
        else:
            print("\n3) Knowledge Graph recommendation stage skipped.")

        # =====================================================
        # 4. External test dataset stage
        # =====================================================
        if RUN_TEST_MODEL:
            print("\n4) External test + project prediction stage started...")

            # IMPORTANT:
            # Use full external employee pool during testing.
            # Do NOT filter by KG shortlist here if you want fair testing.
            test_result = test_model(
                project_id=selected_project_id,
                candidate_employee_ids=None,
            )

            predicted_DataFrame = test_result.predicted_df

            print("   => External test + prediction completed successfully")
            print("   => Total predicted rows:", test_result.total_rows)

            if test_result.accuracy is not None:
                print("\n   => External test metrics:")
                print("      Accuracy :", test_result.accuracy)
                print("      Precision:", test_result.precision)
                print("      Recall   :", test_result.recall)
                print("      F1-score :", test_result.f1)
                print("      Threshold:", test_result.threshold)
                print("      Confusion Matrix:", test_result.confusion_matrix)
                print("\n   => External Classification Report:")
                print(test_result.classification_report)
            else:
                print("   => External metrics could not be computed because labels were unavailable.")
        else:
            print("\n4) External test + project prediction stage skipped.")

        # =====================================================
        # 5. Final recommendation (ML only)
        # =====================================================
        if RUN_FINAL_RECOMMENDATION:
            print("\n5) Final recommendation stage [ML only] started...")

            if predicted_DataFrame is None or predicted_DataFrame.empty:
                print("   => Final recommendation skipped because prediction data is empty.")
            else:
                final_rec = generate_final_recommendation(
                    predicted_DataFrame[
                        predicted_DataFrame["project_id"] == selected_project_id
                    ],
                    top_k=top_k,
                )

                if final_rec is None or final_rec.empty:
                    print("   => Final recommendation is empty. No file saved.")
                else:
                    final_rec.to_excel(FINAL_RECOMMENDED_EXCEL, index=False)
                    print("   => Final recommendation generated successfully")
                    print(f"   => Saved: {FINAL_RECOMMENDED_EXCEL}")
        else:
            print("\n5) Final recommendation stage skipped.")

        # =====================================================
        # 6. Hybrid recommendation (ML + KG)
        # =====================================================
        if RUN_HYBRID_RECOMMENDATION:
            print("\n6) Hybrid recommendation stage [ML + KG] started...")

            if predicted_DataFrame is None or predicted_DataFrame.empty:
                print("   => Hybrid recommendation skipped because ML prediction data is missing.")
            elif graph_rec is None or graph_rec.empty:
                print("   => Hybrid recommendation skipped because KG recommendation data is missing.")
            else:
                hybrid_rec = hybrid_recommendation(
                    predicted_df=predicted_DataFrame,
                    graph_rec=graph_rec,
                    project_id=selected_project_id,
                    top_k=total_number_hybrid,
                )

                if hybrid_rec is None or hybrid_rec.empty:
                    print("   => Hybrid recommendation is empty. No file saved.")
                else:
                    hybrid_rec.to_excel(HYBRID_RECOMMENDED_EXCEL, index=False)
                    print("   => Hybrid recommendation generated successfully")
                    print(f"   => Saved: {HYBRID_RECOMMENDED_EXCEL}")
        else:
            print("\n6) Hybrid recommendation stage skipped.")

        print("\nHAF-EPA Main Controller Finished.\n")


if __name__ == "__main__":
    main()