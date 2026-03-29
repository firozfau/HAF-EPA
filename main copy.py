from __future__ import annotations

from models.generate_train_model import generate_train_model
from models.evaluate_model import evaluate_model
from models.test_model import test_model
from knowledge_graph.kg_recommend import kg_recommend
from models.final_recommendation import generate_final_recommendation
from models.hybrid_recommendation import hybrid_recommendation

from config import FINAL_RECOMMENDED_EXCEL, HYBRID_RECOMMENDED_EXCEL,KNOWLEDGE_RECOMMENDED_EXCEL


# ===== SWITCH =====
RUN_GENERATE_TRAIN_MODEL = False
RUN_EVALUATE_MODEL = False
RUN_TEST_MODEL = False
RUN_KG_RECOMMEND = False
RUN_FINAL_RECOMMENDATION = False
RUN_HYBRID_RECOMMENDATION = False


#Activation and select part
TRAINING_ON = False 

selected_project_id = "P00024"
total_number_hybrid = 100
top_k= 100


def main() -> None:
    print("\nHAF-EPA Main Controller Starting...\n")

    graph_rec = None
    predicted_DataFrame = None
    final_rec = None

    if TRAINING_ON:
        RUN_GENERATE_TRAIN_MODEL = True
        RUN_EVALUATE_MODEL = True
        RUN_TEST_MODEL = False
        RUN_KG_RECOMMEND = False
        RUN_FINAL_RECOMMENDATION = False
        RUN_HYBRID_RECOMMENDATION = False
    else:
        RUN_GENERATE_TRAIN_MODEL = False
        RUN_EVALUATE_MODEL = False
        RUN_TEST_MODEL = True
        RUN_KG_RECOMMEND = True
        RUN_FINAL_RECOMMENDATION = True
        RUN_HYBRID_RECOMMENDATION = True


    if RUN_GENERATE_TRAIN_MODEL:
        generate_train_model()
  
    if RUN_EVALUATE_MODEL:
        evaluate_matrics = evaluate_model() 

        print("Evaluate traing model:")
        print("   => Model Accuracy:", evaluate_matrics.accuracy)
        print("   => Precision:", evaluate_matrics.precision)
        print("   => Recall:", evaluate_matrics.recall)
        print("   => F1-score:", evaluate_matrics.f1)
        print("\nConfusion matrix: ",evaluate_matrics.confusion_matrix) 
        print("\nClassification report:")
        print(" ",evaluate_matrics.classification_report)

    # KG first for hybrid alignment
    if RUN_KG_RECOMMEND:
        graph_rec = kg_recommend()
        graph_rec.to_excel(KNOWLEDGE_RECOMMENDED_EXCEL, index=False)

        print("\n --> Successfully complete knowladge base graph ")
        print(f"    => Successfully generated  Knowledge Graph recommendation: {KNOWLEDGE_RECOMMENDED_EXCEL}") 


    # test_model aligned with selected project + KG candidates
    if RUN_TEST_MODEL:
        predicted_DataFrame = test_model()
        print("\n --> Successfully complete TEST MODEL ")

    if RUN_FINAL_RECOMMENDATION:
        print("\n --> Generating Final Recommendations [Rank only]")
        final_rec = generate_final_recommendation(predicted_DataFrame, top_k=top_k)

        if final_rec.empty:
            print("    => Final recommendation is empty. No file saved.\n")
            return 
        else:
            final_rec.to_excel(FINAL_RECOMMENDED_EXCEL, index=False)
            print(f"    => Successfully generated final recommendation: {FINAL_RECOMMENDED_EXCEL}")

    if RUN_HYBRID_RECOMMENDATION:
        print("\n --> Generating Hybrid Recommendations...")

        if predicted_DataFrame is None or graph_rec is None or final_rec is None:
            print("    => Please set True for, [RUN_TEST_MODEL, RUN_KG_RECOMMEND, RUN_FINAL_RECOMMENDATION, RUN_HYBRID_RECOMMENDATION]" )
            return 
        else:
            hybrid_rec = hybrid_recommendation(predicted_df=predicted_DataFrame,graph_rec=graph_rec, project_id=selected_project_id, top_k=total_number_hybrid, )

            hybrid_rec.to_excel(HYBRID_RECOMMENDED_EXCEL, index=False)
            print(f"    => Successfully generated Hybrid recommendation: {HYBRID_RECOMMENDED_EXCEL}")


if __name__ == "__main__":
    main()