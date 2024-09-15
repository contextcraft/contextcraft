# main.py

from method_name_gen import MethodNameGeneration

if __name__ == "__main__":
    api_key = "YOUR_API_KEY_HERE"
    method_name_gen = MethodNameGeneration(api_key)
    
    # Initialize the model
    method_name_gen.initialize_model()
    
    # Path to the input CSV file and the output CSV file
    input_csv_file = "input_prompts.csv"
    output_csv_file = "output_results.csv"
    
    # Process prompts from the input CSV and generate outputs
    method_name_gen.process_prompts_from_csv(input_csv_file, output_csv_file)
