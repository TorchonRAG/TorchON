# from abidlabs : https://huggingface.co/spaces/abidlabs/GPT-Baker/blob/main/maker.py

import gradio as gr
import requests
import json
import huggingface_hub
from huggingface_hub import HfApi
import os

HF_TOKEN = os.environ["HF_TOKEN"]
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

zephyr_7b_beta = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta/"


welcome_message = """
Hi! I'll help you **build a GPT**. You can say something like, "make a bot that gives advice on how to grow your startup."
What would you like to make?
"""

welcome_preview_message = """
Welcome to **{}**! Say something like: 
"{}"
"""

# sample_response = """
# Certainly! Here we go:

# Title: Recipe Recommender
# System Prompt: Utilize your language model abilities to suggest delicious recipes based on user preferences such as ingredients, cuisine type, cooking time, etc. Ensure accuracy and variety while maintaining a conversational style with the user. 
# Example User Input: Vegetarian dinner ideas under 30 minutes
# """

zephyr_system_prompt = """
You are an AI whose job it is to help users create their own chatbots. In particular, you need to respond succintly in a friendly tone, write a system prompt for an LLM, a catchy title for the chatbot, and a very short example user input. Make sure each part is included.
For example, if a user says, "make a bot that gives advice on how to grow your startup", first do a friendly response, then add the title, system prompt, and example user input. Immediately STOP after the example input. It should be EXACTLY in this format:
Sure, I'd be happy to help you build a bot! I'm generating a title, system prompt, and an example input. How do they sound? Feel free to give me feedback!
Title: Startup Coach
System prompt: Your job as an LLM is to provide good startup advice. Do not provide extraneous comments on other topics. Be succinct but useful. 
Example input: Risks of setting up a non-profit board
Here's another example. If a user types, "Make a chatbot that roasts tech ceos", respond: 
Sure, I'd be happy to help you build a bot! I'm generating a title, system prompt, and an example input. How do they sound? Feel free to give me feedback!
Title: Tech Roaster
System prompt: As an LLM, your primary function is to deliver hilarious and biting critiques of technology CEOs. Keep it witty and entertaining, but also make sure your jokes aren't too mean-spirited or factually incorrect. 
Example input: Elon Musk
"""

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


def predict_beta(message, chatbot=[], system_prompt=zephyr_system_prompt):
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


def extract_title_prompt_example(text, title, system_prompt, example_input):
    try:
        # Finding the indices of the key terms
        text_start = text.rfind("<|assistant|>", ) + len("<|assistant|>")
        text = text[text_start:]
    except ValueError:
        pass
    try:
        title_start = text.lower().rfind("title:") + len("title:")    
        prompt_start = text.lower().rfind("system prompt:")
        title = text[title_start:prompt_start].strip()
    except ValueError:
        pass
    try:
        prompt_start = text.lower().rfind("system prompt:") + len("system prompt:")
        example_start = text.lower().rfind("example input:")
        system_prompt = text[prompt_start:example_start].strip()
    except ValueError:
        pass
    try:
        example_start = text.lower().rfind("example input:") + len("example input:")
        example_input = text[example_start:].strip()
        example_input = example_input[:example_input.index("\n")]
    except ValueError:
        pass
    return text, title, system_prompt, example_input

def make_open_gpt(message, history, current_title, current_system_prompt, current_example_input):
    response = predict_beta(message, history, zephyr_system_prompt)
    response, title, system_prompt, example_input = extract_title_prompt_example(response, current_title, current_system_prompt, current_example_input)
    return "", history + [(message, response)], title, system_prompt, example_input, [(None, welcome_preview_message.format(title, example_input))], example_input, gr.Column(visible=True), gr.Group(visible=True)

def set_title_example(title, example):
    return [(None, welcome_preview_message.format(title, example))], example, gr.Column(visible=True), gr.Group(visible=True)

chatbot_preview = gr.Chatbot(layout="panel")
textbox_preview = gr.Textbox(scale=7, container=False)

def test_preview_chatbot(message, history, system_prompt):
    response = predict_beta(message, history, system_prompt)
    text_start = response.rfind("<|assistant|>", ) + len("<|assistant|>")
    response = response[text_start:]
    return response


def strip_invalid_filename_characters(filename: str, max_bytes: int = 200) -> str:
    """Strips invalid characters from a filename and ensures that the file_length is less than `max_bytes` bytes."""
    filename = filename.replace(" ", "-")
    filename = "".join([char for char in filename if char.isalnum() or char in "_-"])
    filename_len = len(filename.encode())
    if filename_len > max_bytes:
        while filename_len > max_bytes:
            if len(filename) == 0:
                break
            filename = filename[:-1]
            filename_len = len(filename.encode())
    return filename


constants = """
SYSTEM_PROMPT = "{}"
TITLE = "{}"
EXAMPLE_INPUT = "{}"
"""


def publish(textbox_system_prompt, textbox_title, textbox_example, textbox_token):
    source_file = 'app_template.py'
    destination_file = 'app.py'
    constants_formatted = constants.format(textbox_system_prompt, textbox_title, textbox_example)
    with open(source_file, 'r') as file:
        original_content = file.read()
    with open(destination_file, 'w') as file:
        file.write(constants_formatted + original_content)
    title = strip_invalid_filename_characters(textbox_title, max_bytes=30)
    api = HfApi(token=textbox_token)
    new_space = api.create_repo(
        repo_id=f"open-gpt-{title}",
        repo_type="space",
        exist_ok=True,
        private=False,
        space_sdk="gradio",
        token=textbox_token,
    )
    api.upload_file(
        repo_id=new_space.repo_id,
        path_or_fileobj='app.py',
        path_in_repo='app.py',
        token=textbox_token,
        repo_type="space",
    )
    api.upload_file(
        repo_id=new_space.repo_id,
        path_or_fileobj='README_template.md',
        path_in_repo='README.md',
        token=textbox_token,
        repo_type="space",
    )
    huggingface_hub.add_space_secret(
        new_space.repo_id, "HF_TOKEN", textbox_token, token=textbox_token
    )

    return gr.Markdown(f"Published to https://huggingface.co/spaces/{new_space.repo_id} ✅", visible=True), gr.Button("Publish", interactive=True)
    
    
css = """
#preview-tab-button{
    font-weight: bold;
}
"""

with gr.Blocks(css=css) as demo:
    gr.Markdown("🥧 **GPT Baker** lets you create your own **open-source GPTs**. Start chatting below to automatically bake your GPT (or you can manually configure the recipe in the second tab). You can build and test them for free, but will need a [HF Pro account](https://huggingface.co/subscribe/pro) to publish them on Spaces (as Open GPTs are powered by the Zephyr 7B beta model using the HF Inference API). You will **not be charged** for usage of your Open GPT as the HF Inference API Pro membership does not charge per-query. Find your token here: https://huggingface.co/settings/tokens")
    with gr.Row():
        with gr.Column(scale=3):
            with gr.Tab("Create"):
                chatbot_maker = gr.Chatbot([(None, welcome_message)], layout="panel", elem_id="chatbot-maker")
                with gr.Group():
                    with gr.Row():
                        textbox_maker = gr.Textbox(placeholder="Make a bot that roasts tech CEOs", scale=7, container=False, autofocus=True)
                        submit_btn = gr.Button("Bake 👩‍🍳", variant="secondary")
            with gr.Tab("Configure Recipe"):
                textbox_title = gr.Textbox("GPT Preview", label="Title")
                textbox_system_prompt = gr.Textbox(label="System prompt", lines=6)
                textbox_example = gr.Textbox(label="Placeholder example", lines=2)
            with gr.Tab("Files"):
                gr.Markdown("RAG coming soon!")
        with gr.Column(visible=False, scale=5) as preview_column:
            with gr.Tab("🪄 Preview of your Open GPT", elem_id="preview-tab") as preview_tab:
                gr.ChatInterface(test_preview_chatbot, chatbot=chatbot_preview, textbox=textbox_preview, autofocus=False, submit_btn="Test", additional_inputs=[textbox_system_prompt])
    with gr.Group(visible=False) as publish_row:
        with gr.Row():
            textbox_token = gr.Textbox(show_label=False, placeholder="Ready to publish to Spaces? Enter your HF token here", scale=7)
            publish_btn = gr.Button("Publish", variant="primary")

    published_status = gr.Markdown(visible=False)
    
    gr.on([submit_btn.click, textbox_maker.submit], make_open_gpt, [textbox_maker, chatbot_maker, textbox_title, textbox_system_prompt, textbox_example], [textbox_maker, chatbot_maker, textbox_title, textbox_system_prompt, textbox_example, chatbot_preview, textbox_preview, preview_column, publish_row])
    gr.on([textbox_title.blur, textbox_example.blur], set_title_example, [textbox_title, textbox_example], [chatbot_preview, textbox_preview, preview_column, publish_row])

    publish_btn.click(lambda : gr.Button("Publishing...", interactive=False), None, publish_btn).then(publish, [textbox_system_prompt, textbox_title, textbox_example, textbox_token], [published_status, publish_btn])

demo.launch()