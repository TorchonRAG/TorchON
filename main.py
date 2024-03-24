# .main.py

import dspy
from dsp.modules.anthropic import Claude
from dsp.modules.google import Google
# from dsp.utils import format_examples
# from dspy.retrieve.chromadb_rm import ChromadbRM
from dspy.evaluate import Evaluate
# from .datasets.hotpotqa import HotPotQA
from dspy.teleprompt import BootstrapFewShotWithRandomSearch, BootstrapFinetune
from dsp.modules.lm import LM
from dsp.utils.utils import deduplicate
import huggingface_hub
from huggingface_hub import HfApi
import os
import random
import requests
from pathlib import Path
from typing import Optional, Any, List, Dict
import base64
import time

import chromadb
from chromadb.utils import embedding_functions
from functools import wraps
from dotenv import load_dotenv, set_key
from pydantic import BaseModel
import gradio as gr
from src.dataloaders.dataloader import DataProcessor, DocumentLoader
from src.torchonlongform.longform import LongFormContent, PromptToExample, PromptToRetrieval , Retriever , LongFormQA, LongFormQAWithAssertions
from src.torchonsyntheticdata.syntheticdata import SyntheticDataGenerator, SyntheticDataHandler
from src.config.config import APIKeyManager
from src.torchonrag.torchonrag import Upsert, MyRetriever
from src.torchonpublish.publish import TorchonPublisher
# from src.torchonapplicationcompiler.torchoncompiler import 
from src.torchonllms.torchonrequestclients import ClaudeGenerate
from src.torchonragmaker.torchonragmaker import CreateRAG
import logging

kwargs = {
    "temperature": 0.7,
    "max_tokens": 2048,
    "top_p": 0.9,
    "top_k": 40,
    "n": 1, 
}

class Application:
    def __init__(self):

        # For publishing  - add these prompts to .env file, then load them accordingly !
        self.title = ""
        self.system_prompt = ""
        self.example_input = ""
        self.hf_token = ""
        self.anthropic_api_key = ""
        self.openai_api_key = ""
        self.Retriever = MyRetriever(api_key=self.anthropic_api_key)
        self.api_key_manager = APIKeyManager()
        self.data_processor = DataProcessor(source_file="", collection_name="torchon-tonic-ai", persist_directory="/your_files_here")
#       self.claude_model_manager = ClaudeModelManager()
        self.synthetic_data_handler = SyntheticDataHandler()
        self.CreateRAG = CreateRAG()
#       self.ChatbotManager = ChatbotManager
        # history = history[]
        # self.handle_chatbot_interaction()

    def set_api_keys(self, anthropic_api_key, openai_api_key, hf_token):
        return self.api_key_manager.set_api_keys(anthropic_api_key, openai_api_key, hf_token)

    def handle_file_upload(self, uploaded_file):
        self.data_processor.source_file = uploaded_file.name
        loaded_data = self.data_processor.load_data_from_source_and_store()
        print("Data from {uploaded_file.name} loaded and stored successfully.")
        return loaded_data

    def handle_synthetic_data(self, schema_class_name, sample_size):  # schema_class_name
        synthetic_data = self.synthetic_data_handler.generate_data(sample_size=int(sample_size))
        synthetic_data_str = "\n".join([str(data) for data in synthetic_data])
        print ("Generated {sample_size} synthetic data items:\n{synthetic_data_str}")
        return synthetic_data

    def generate_content(prompt):
        content_generator = LongFormContent()
        result = content_generator(prompt)
        return result.blog

    def handle_chatbot_interaction(self, text, model_select, top_p, temperature, repetition_penalty, max_length_tokens, max_context_length_tokens):
        chatbot_response = self.Retriever.generate_response(text, None, model_select, top_p, temperature, repetition_penalty, max_length_tokens, max_context_length_tokens)
        return chatbot_response
    # def generate_RAG(self, input_text, title, system_prompt, example_input):
    #     response = self.CreateRAG.create_rag(input_text, title, system_prompt, example_input)
    #     return response

    def publish(self):
        publisher = TorchonPublisher(self.title, self.hf_token, self.anthropic_api_key)
        return publisher.publish()

    def main(self):
        with gr.Blocks() as demo:
            with gr.Accordion("API Keys", open=True) as api_keys_accordion:
                with gr.Row():
                    anthropic_api_key_input = gr.Textbox(label="Anthropic API Key", type="password")
                    openai_api_key_input = gr.Textbox(label="OpenAI API Key", type="password")
                    hf_token_input = gr.Textbox(type="password", label="Hugging Face Write Token")
                    github_token_input = gr.Textbox(type="password", label="GitHub Write Token")

                submit_button = gr.Button("Submit")
                confirmation_output = gr.Textbox(label="Confirmation", visible=False)

                submit_button.click(
                    fn=self.set_api_keys,
                    inputs=[anthropic_api_key_input, openai_api_key_input, hf_token_input, github_token_input],
                    outputs=confirmation_output
                )

            with gr.Accordion("Upload Data") as upload_data_accordion:
                file_upload = gr.File(label="Upload Data Files")
                folder_upload = gr.Files(label="Upload Folder")
                webpage_input = gr.Textbox(label="Web Page URL")
                github_input = gr.Textbox(label="GitHub Repository Link")
                    
                file_upload_button = gr.Button("Process Data")
                validation_output = gr.JSON(label="Validation Output")
                        
                file_upload_button.click(
                    fn=self.handle_file_upload,
                    inputs=[file_upload, folder_upload, webpage_input, github_input],
                    outputs=validation_output
                )

            with gr.Accordion("Generate Synthetic Data") as generate_data_accordion:
                schema_input = gr.Textbox(label="Schema Class Name")
                sample_size_input = gr.Number(label="Sample Size", value=100)
                synthetic_data_button = gr.Button("Generate Synthetic Data")
                synthetic_data_output = gr.Textbox()

                synthetic_data_button.click(
                    fn=self.handle_synthetic_data,
                    inputs=[schema_input, sample_size_input],
                    outputs=synthetic_data_output
                )

            with gr.Accordion("SimpleTestingChatbot"):
                with gr.Tab("Create"):
                    chatbot_maker = gr.Chatbot(layout="panel", elem_id="chatbot-maker")
                    textbox_maker = gr.Textbox(placeholder="Your words here", autofocus=True)
                    submit_btn = gr.Button("Send")

                submit_btn.click(
                    fn=self.CreateRAG,
                    inputs=[textbox_maker],
                    outputs=[chatbot_maker]
                )

                with gr.Tab("Publish"):
                    title_input = gr.Textbox(value=self.title, label="Title")
                    system_prompt_input = gr.Textbox(value=self.system_prompt, label="System Prompt")
                    example_input = gr.Textbox(value=self.example_input, label="Example Input")
                    hf_token_input = gr.Textbox(label="Hugging Face Token", type="password")
                    publish_btn = gr.Button("Publish")

                    publish_btn.click(
                        fn=self.publish,
                        inputs=[title_input, system_prompt_input, example_input, hf_token_input],
                        outputs=[]
                    )

        demo.launch()

if __name__ == "__main__":
    app = Application()
    app.main()