# main.py

from method_name_pred import MethodNamePrediction

if __name__ == "__main__":
    api_key = "YOUR_OPENAI_API_KEY_HERE"
    method_name_pred = MethodNamePrediction(api_key)
    
    # Path to the input CSV file and the output CSV file
    input_csv_file = "input_prompts.csv"
    output_csv_file = "output_results.csv"
    
    # Process prompts from the input CSV and generate outputs
    method_name_pred.process_prompts_from_csv(input_csv_file, output_csv_file)
