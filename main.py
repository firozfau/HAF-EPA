from data_loader.load_datasets import load_datasets
from models.train_model import initialize_training_pipeline


def main() -> None:
    print("HAF-EPA — Week 1 Setup")
    print("=" * 40)

    dataset_info = load_datasets()
    training_info = initialize_training_pipeline()

    print("\nDataset Loader Status:")
    print(dataset_info)

    print("\nTraining Pipeline Status:")
    print(training_info)

    print("\nWeek 1 Deliverables Completed:")
    print("- Project structure created")
    print("- Initial dataset loading module prepared")
    print("- Initial model training module prepared")
    print("- Main entry file prepared")
    print("- Dependencies documented")

    print("\nNext Step:")
    print("Proceed to Week 2: full dataset loading and validation.")


if __name__ == "__main__":
    main()
