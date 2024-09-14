# csv_handler.py

import pandas as pd

class CSVHandler:
    @staticmethod
    def read_prompts_from_csv(csv_file_path):
        """Reads a CSV file containing prompts and returns it as a DataFrame."""
        prompts_df = pd.read_csv(csv_file_path)
        if 'prompt' not in prompts_df.columns:
            raise ValueError("The CSV file must contain a 'prompt' column")
        return prompts_df

    @staticmethod
    def save_output_to_csv(dataframe, output_file_path):
        """Saves the DataFrame containing prompts and outputs to a CSV file."""
        dataframe.to_csv(output_file_path, index=False)
        print(f"Output saved to {output_file_path}")
