# method_name_gen.py

from model_config import ModelConfiguration
from text_processing import TextProcessing
from csv_handler import CSVHandler

class MethodNameGeneration:
    def __init__(self, api_key):
        """Initializes the MethodNameGeneration class with the provided API key."""
        self.model_config = ModelConfiguration(api_key)

    def initialize_model(self):
        """Sets up the model by calling the ModelConfiguration class."""
        self.model_config.create_model()

    def process_prompts_from_csv(self, input_csv_file, output_csv_file):
        """Processes the prompts from a CSV file and generates outputs."""
        # Read the prompts from the CSV file
        prompts_df = CSVHandler.read_prompts_from_csv(input_csv_file)

        # Generate output for each prompt
        text_processor = TextProcessing(self.model_config.get_chat_session())
        prompts_df['output'] = prompts_df['prompt'].apply(text_processor.generate_output_for_prompt)

        # Save the result to an output CSV file
        CSVHandler.save_output_to_csv(prompts_df, output_csv_file)
