from difflib import SequenceMatcher
# LFM class
class LLMBasedFeedback:
    def __init__(self, api_client):
        self.api_client = api_client  # Placeholder for an LLM API client

    def predict_method_name(self, functional_description):
        """Predict method name using LLM API."""
        response = self.api_client.generate_method_name(functional_description)
        return response['predicted_method_name']  # Adjust based on actual API response structure

    def compare_with_actual(self, predicted_name, actual_name):
        """Compare predicted method name with actual using edit distance."""
        return SequenceMatcher(None, predicted_name, actual_name).ratio()

    def generate_feedback(self, predicted_name, actual_name):
        """Generate feedback statement comparing predicted and actual method names."""
        edit_distance_score = self.compare_with_actual(predicted_name, actual_name)
        feedback = (
            f"The predicted method name by base LLM: '{predicted_name}' "
            f"which has an edit distance score: {edit_distance_score:.2f} as compared to "
            f"the actual method name: '{actual_name}'."
        )
        return feedback
