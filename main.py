from __future__ import annotations

from models.generate_train_model import generate_train_model
from models.evaluate_model import evaluate_model
from models.test_model import test_model
from knowledge_graph.kg_recommend import kg_recommend
from models.final_recommendation import generate_final_recommendation
from models.hybrid_recommendation import hybrid_recommendation

from config import FINAL_RECOMMENDED_EXCEL, HYBRID_RECOMMENDED_EXCEL


# ===== SWITCH =====
RUN_GENERATE_TRAIN_MODEL = False
RUN_EVALUATE_MODEL = False
RUN_TEST_MODEL = True
RUN_KG_RECOMMEND = True
RUN_FINAL_RECOMMENDATION = True
RUN_HYBRID_RECOMMENDATION = True

# Requested project for hybrid recommendation
selected_project_id = "P00024"


def main() -> None:
    print("HAF-EPA Main Controller Starting...\n")

    graph_rec = None
    predicted_DataFrame = None
    final_rec = None

    # 1. Train model
    if RUN_GENERATE_TRAIN_MODEL:
        generate_train_model()

    # 2. Evaluate model
    if RUN_EVALUATE_MODEL:
        evaluate_model()

    # 3. Test model
    if RUN_TEST_MODEL:
        predicted_DataFrame = test_model()
        print("\npredicted_DataFrame:--->")
        print(predicted_DataFrame)

    # 4. Knowledge Graph recommendation
    if RUN_KG_RECOMMEND:
        graph_rec = kg_recommend()

    # 5. Final recommendation from ML only
    if RUN_FINAL_RECOMMENDATION:
        print("\nGenerating Final Recommendations [Rank only]")

        if predicted_DataFrame is None:
            print("Final recommendation skipped: predicted_DataFrame not available.")
            print("Please set RUN_TEST_MODEL = True and RUN_FINAL_RECOMMENDATION = True")
        else:
            final_rec = generate_final_recommendation(predicted_DataFrame, top_k=5)
            final_rec.to_excel(FINAL_RECOMMENDED_EXCEL, index=False)
            print(f"\nSuccessfully generated final recommendation: {FINAL_RECOMMENDED_EXCEL}")


    # 6. Hybrid recommendation
    if RUN_HYBRID_RECOMMENDATION:
        print("\nGenerating Hybrid Recommendations...")

        if predicted_DataFrame is None or graph_rec is None:
            print("Please set True for [RUN_TEST_MODEL, RUN_KG_RECOMMEND, RUN_FINAL_RECOMMENDATION, RUN_HYBRID_RECOMMENDATION]")
        else:

            ml_projects = set(predicted_DataFrame["project_id"].dropna().unique())
            kg_projects = set(graph_rec["project_id"].dropna().unique())

            common_projects = sorted(list(ml_projects.intersection(kg_projects)))

            print(f"\nRequested project_id: {selected_project_id}")

            if selected_project_id not in common_projects:
                print(f"\nProject {selected_project_id} is not available in both ML and KG outputs.")

                if common_projects:
                    print("\nAvailable common project_ids:")
                    print(common_projects[:20])
                else: 
                    print("No common project_id found, Hybrid recommendation skipped.")
                    return
            
             # Generate hybrid
            hybrid_rec = hybrid_recommendation(predicted_df=predicted_DataFrame, graph_rec=graph_rec, project_id=selected_project_id,  top_k=5,)
 
            if hybrid_rec.empty or hybrid_rec["graph_score"].sum() == 0:
                print(f"\n Hybrid approach NOT suitable for project {selected_project_id}")
                print("Reason: graph_score = 0 (no overlap between ML and KG)\n")

               
                all_graph_rec = graph_rec.copy()
                all_project_ids = predicted_DataFrame["project_id"].dropna().unique()
                found_project_id = None

                for project_id in all_project_ids:
                    project_graph_rec = all_graph_rec[all_graph_rec["project_id"] == project_id  ].copy()

                    hybrid_rec = hybrid_recommendation(predicted_df=predicted_DataFrame, graph_rec=project_graph_rec,  project_id=project_id, top_k=5, )

                    if not hybrid_rec.empty and hybrid_rec["graph_score"].sum() != 0:
                        found_project_id = project_id
                        print(f"First suitable project_id for hybrid: {project_id}")
                        break

            if found_project_id is None:
                print("No suitable project_id found for hybrid recommendation.")


            else:
                hybrid_rec.to_excel(HYBRID_RECOMMENDED_EXCEL, index=False)

                print(f"\nSuccessfully generated Hybrid recommendation: {HYBRID_RECOMMENDED_EXCEL}")
                print("\nHybrid recommendation sample:")
                print(hybrid_rec.head())


if __name__ == "__main__":
    main()