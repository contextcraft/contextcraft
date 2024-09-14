from data_load import DataLoad
from UniXcoder_FineTune import UniXcoder_FineTune
from UniXcoder_Test import UniXcoder_Test
import pandas as pd

def main():
    # File paths for Python-specific dataset
    train_file = "/mnt/data/python_train.csv"
    test_file = "/mnt/data/python_test.csv"
    
    # Load and preprocess the data
    data_loader = DataLoad(train_file, test_file)
    train_df, test_df = data_loader.load_data()

    # Fine-tune the UniXcoder model
    fine_tuner = UniXcoder_FineTune(num_labels=len(train_df['Method Name'].unique()))
    fine_tuned_model = fine_tuner.fine_tune(train_df, test_df)

    # Test the fine-tuned model and predict method names
    tester = UniXcoder_Test('./unixcoder-methodname')
    test_df = tester.evaluate(test_df)
    
    # Save the updated test data with predictions to a new CSV file
    output_file = "/mnt/data/python_test_with_predictions_unixcoder.csv"
    test_df.to_csv(output_file, index=False)
    print(f"Predicted method names saved to {output_file}")

if __name__ == "__main__":
    main()
