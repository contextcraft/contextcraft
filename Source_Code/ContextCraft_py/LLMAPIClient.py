import openai
#class Large Language Model API
class LLMAPIClient:
    def __init__(self, api_key):
        openai.api_key = api_key

    def predict_method_name(self, func_desc):
        """Use OpenAI's API to predict the method name from a functional description."""
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant trained to generate method names for functional descriptions of programming tasks."},
                {"role": "user", "content": f"Generate a method name for the following functional description: {func_desc}"}
            ]
        )
        method_name = response.choices[0].message['content'].strip()
        return method_name

    def generate_feedback(self, predicted_name, actual_name):
        """Generate feedback comparing the predicted method name with the actual method name."""
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant trained to provide feedback on programming method names."},
                {"role": "user", "content": f"Provide feedback on how well the predicted method name '{predicted_name}' matches the actual method name '{actual_name}'."}
            ]
        )
        feedback = response.choices[0].message['content'].strip()
        return feedback
