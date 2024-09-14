# model_config.py

class ModelConfiguration:
    def __init__(self, api_key):
        import google.generativeai as genai
        self.api_key = api_key
        genai.configure(api_key=self.api_key)
        self.model = None
        self.chat_session = None

    def create_model(self, temperature=0.9, top_p=1, max_output_tokens=2048):
        """Creates and configures the model with the provided parameters."""
        generation_config = {
            "temperature": temperature,
            "top_p": top_p,
            "max_output_tokens": max_output_tokens,
            "response_mime_type": "text/plain",
        }
        self.model = genai.GenerativeModel(
            model_name="gemini-1.0-pro",
            generation_config=generation_config
        )
        self.chat_session = self.model.start_chat(history=[])
    
    def get_chat_session(self):
        """Returns the active chat session."""
        return self.chat_session
