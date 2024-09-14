from data_load import DataLoad
from CodeBERT_FineTune import CodeBERT_FineTune
from CodeBERT_Test import CodeBERT_Test

def main():
    # File paths
    train_file = "/mnt/data/java_train.csv"
    test_file = "/mnt/data/java_test.csv"
    
    # Load and preprocess the data
    data_loader = DataLoad(train_file, test_file)
    train_df, test_df = data_loader.load_data()

    # Fine-tune the CodeBERT model
    fine_tuner = CodeBERT_FineTune(num_labels=len(train_df['Method Name'].unique()))
    fine_tuned_model = fine_tuner.fine_tune(train_df, test_df)

    # Test the fine-tuned model
    tester = CodeBERT_Test('./codebert-methodname')
    accuracy, result_df = tester.evaluate(test_df)
    
    # Display results
    print(f"Accuracy on the test set: {accuracy * 100:.2f}%")
    print(result_df.head())

if __name__ == "__main__":
    main()
