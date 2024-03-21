# This app was made using https://github.com/tonic-ai/torchon

import gradio as gr
import os
import requests
from llama_index.vector_stores.chroma import ChromaVectorStore

# zephyr_7b_beta = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta/"

HF_TOKEN = os.getenv("HF_TOKEN")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
# HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}
systemprompt = os.getenv("SYSTEM_PROMPT")

class ClaudeModelManager:
    def __init__(self, model: str = "claude-3-opus-20240229", api_key: Optional[str] = None, api_base: Optional[str] = None,  **kwargs):
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.api_base = api_base
        self.kwargs = kwargs
        self.history = []  # Track request/response history if needed
        self.initialize_claude(**kwargs)

    def initialize_claude(self, **kwargs):
        print("Initializing Claude...")
        try:
            from anthropic import Anthropic, RateLimitError
            print("Successfully imported Anthropics's API client.")
        except ImportError as err:
            print("Failed to import Anthropics's API client.")
            raise ImportError("Claude requires `pip install anthropic`.") from err

        if not self.api_key:
            raise ValueError("API key is not set. Please ensure it's provided or set in the environment variables.")

        self.client = Anthropic(api_key=self.api_key, model=self.model, **kwargs)
        print("Anthropic client initialized with model:", self.model)

# # Example usage
# kwargs = {
#     "temperature": 0.7,
#     "max_tokens": 2048,
#     "top_p": 0.9,
#     "top_k": 40,
#     "n": 1,  # Number of generations
#     # Add other parameters as needed
# }
# claude_manager = ClaudeModelManager(api_key="ANTHROPIC_API_KEY", **kwargs)    
class ClaudeGenerate(self,  systemprompt:Optional[str] = systemprompt, **kwargs):
    def __init__(model="claude-3-opus-20240229", api_key= ANTHROPIC_API_KEY ):
        self.model = model
        self.api_key = api_key
        self.provider = "default"
        self.base_url = "https://api.anthropic.com/v1/messages"
        self.systemprompt = systemprompt
    def basic_request(self, prompt: str, **kwargs):
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "messages-2023-12-15",
            "content-type": "application/json"
            }

        data = {
            **kwargs,
            "model": self.model,
            "system" = self.systemprompt,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        print("Generating Using Claude...")
        response = requests.post(self.base_url, headers=headers, json=data)
        response = response.json()

        self.history.append({
            "prompt": prompt,
            "response": response,
            "kwargs": kwargs,
        })
        return response

    def __call__(self, prompt, only_completed=True, return_sorted=False, **kwargs):
        response = self.request(prompt, **kwargs)
        completions = [result["text"] for result in response["content"]]

        return completions

def format_prompt(prompt: str, context: str)
    formatted_prompt = "Context :\n {context}\n \nQuestion :\n{prompt}"
    return formatted_prompt
# def predict_beta(message, chatbot=[], system_prompt=""):
#     input_prompt = build_input_prompt(message, chatbot, system_prompt)
#     data = {
#         "inputs": input_prompt
#     }

#     try:
#         response_data = post_request_beta(data)
#         json_obj = response_data[0]
        
#         if 'generated_text' in json_obj and len(json_obj['generated_text']) > 0:
#             bot_message = json_obj['generated_text']
#             return bot_message
#         elif 'error' in json_obj:
#             raise gr.Error(json_obj['error'] + ' Please refresh and try again with smaller input prompt')
#         else:
#             warning_msg = f"Unexpected response: {json_obj}"
#             raise gr.Error(warning_msg)
#     except requests.HTTPError as e:
#         error_msg = f"Request failed with status code {e.response.status_code}"
#         raise gr.Error(error_msg)
#     except json.JSONDecodeError as e:
#         error_msg = f"Failed to decode response as JSON: {str(e)}"
#         raise gr.Error(error_msg)



class ChromaRetriever:
    def initialize_vectorstore(path="/vectorstore")
        db = chromadb.PersistentClient(path=path)
        chroma_collection = db.get_or_create_collection("torchon-tonic-ai")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        return vector_store
    # index = VectorStoreIndex.from_vector_store(
    #     vector_store,
    #     embed_model=embed_model,
    # )
    def query_data(vector_store, query="what's tonic-ai")
        """
        Query Data from the persisted index
        """
        query_engine = vector_store.as_query_engine()
        response = query_engine.query(query)
        return response

def test_preview_chatbot(message, history):
    """
    with message , retrieve context , format prompt and pass message to claude 
    """
    vectorstore = ChromaRetriever.initialize_vectorstore
    context = ChromaRetriever.query_data(vectorstore, message)
    formatted_prompt = format_prompt(message, history, context)
    response = ClaudeGenerate.basic_request(formatted_prompt)
#   response = ClaudeGenerate.basic_request(message, history, systemprompt)
#   text_start = response.rfind("<|assistant|>", ) + len("<|assistant|>")
#   response = response[text_start:]
    return response


welcome_preview_message = f"""
Welcome to **{TITLE}**! Say something like: 
"{EXAMPLE_INPUT}"
"""

chatbot_preview = gr.Chatbot(layout="panel", value=[(None, welcome_preview_message)])
textbox_preview = gr.Textbox(scale=7, container=False, value=EXAMPLE_INPUT)

demo = gr.ChatInterface(test_preview_chatbot, chatbot=chatbot_preview, textbox=textbox_preview)

demo.launch()