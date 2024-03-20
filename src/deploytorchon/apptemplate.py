# from https://huggingface.co/spaces/abidlabs/GPT-Baker/blob/main/app_template.py

import gradio as gr
import os
import requests

zephyr_7b_beta = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta/"

HF_TOKEN = os.getenv("HF_TOKEN")
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

def build_input_prompt(message, chatbot, system_prompt):
    """
    Constructs the input prompt string from the chatbot interactions and the current message.
    """
    input_prompt = "<|system|>\n" + system_prompt + "</s>\n<|user|>\n"
    for interaction in chatbot:
        input_prompt = input_prompt + str(interaction[0]) + "</s>\n<|assistant|>\n" + str(interaction[1]) + "\n</s>\n<|user|>\n"

    input_prompt = input_prompt + str(message) + "</s>\n<|assistant|>"
    return input_prompt


def post_request_beta(payload):
    """
    Sends a POST request to the predefined Zephyr-7b-Beta URL and returns the JSON response.
    """
    response = requests.post(zephyr_7b_beta, headers=HEADERS, json=payload)
    response.raise_for_status()  # Will raise an HTTPError if the HTTP request returned an unsuccessful status code
    return response.json()


def predict_beta(message, chatbot=[], system_prompt=""):
    input_prompt = build_input_prompt(message, chatbot, system_prompt)
    data = {
        "inputs": input_prompt
    }

    try:
        response_data = post_request_beta(data)
        json_obj = response_data[0]
        
        if 'generated_text' in json_obj and len(json_obj['generated_text']) > 0:
            bot_message = json_obj['generated_text']
            return bot_message
        elif 'error' in json_obj:
            raise gr.Error(json_obj['error'] + ' Please refresh and try again with smaller input prompt')
        else:
            warning_msg = f"Unexpected response: {json_obj}"
            raise gr.Error(warning_msg)
    except requests.HTTPError as e:
        error_msg = f"Request failed with status code {e.response.status_code}"
        raise gr.Error(error_msg)
    except json.JSONDecodeError as e:
        error_msg = f"Failed to decode response as JSON: {str(e)}"
        raise gr.Error(error_msg)

def test_preview_chatbot(message, history):
    response = predict_beta(message, history, SYSTEM_PROMPT)
    text_start = response.rfind("<|assistant|>", ) + len("<|assistant|>")
    response = response[text_start:]
    return response


welcome_preview_message = f"""
Welcome to **{TITLE}**! Say something like: 
"{EXAMPLE_INPUT}"
"""

chatbot_preview = gr.Chatbot(layout="panel", value=[(None, welcome_preview_message)])
textbox_preview = gr.Textbox(scale=7, container=False, value=EXAMPLE_INPUT)

demo = gr.ChatInterface(test_preview_chatbot, chatbot=chatbot_preview, textbox=textbox_preview)

demo.launch()