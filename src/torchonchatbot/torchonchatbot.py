# ./src/torchonchatbot/torchonchatbot.py

import logging
import os
import time
from typing import List
from main import CreateRAG
from main import TitleGenerator
from main import MyRetriever
from main import Conversation
from main import SeparatorStyle
from gradio_client import gr

import logging
import os
import time
from typing import List
from  main import CreateRAG
import chromadb
from chromadb.utils import embedding_functions
class ChatbotManager:
    def __init__(self, model_name):
        self.model = model_name
        self.history = []
        self.title = ""
        self.system_prompt = ""
        self.example_input = ""
        self.hf_token = ""
        self.logger = logging.getLogger("gradio_logger")
        self.TitleGenerator = TitleGenerator()
        self.CreateRAG = CreateRAG.create_rag()
    def make_open_gpt(self, message, current_title, current_system_prompt, current_example_input):
        response = self.generate_response(message, self.history, current_system_prompt)
        system_prompt = self.CreateRAG(message)
        example_input = self.CreateRAG(message)
        title = self.TitleGenerator(response)
        self.history.append((message, response))
        self.title = title
        self.system_prompt = system_prompt
        self.example_input = example_input
        return "", self.history, title, system_prompt, example_input

    def set_title_example(self, title, example):
        self.title = title
        self.example_input = example
        return title, example


    def generate_response(self, text, top_p=0.9, temperature=0.7, repetition_penalty=1.0, max_length_tokens=2048, max_context_length_tokens=1024):

        kwargs = {
            "temperature": temperature,
            "max_tokens": max_length_tokens,
            "top_p": top_p,
            "top_k": 40,
            "n": 1,
            "repetition_penalty": repetition_penalty,
            "model": self.model,  
        }

        prompt = self.generate_prompt_with_history(text, self.history, max_length=max_context_length_tokens)
        if prompt is None:
            return "Error: Prompt generation failed.", self.history, "Generate: Failed"

        gradio_chatbot_output = MyRetriever.retrieve_passages(query=prompt)
        self.history.append((text, gradio_chatbot_output))  

        return gradio_chatbot_output, self.history, "Generate: Success"

    def configure_logger(self):
        logger = logging.getLogger("gradio_logger")
        logger.setLevel(logging.DEBUG)

        timestr = time.strftime("%Y%m%d-%H%M%S")
        os.makedirs("logs", exist_ok=True)
        file_handler = logging.FileHandler(
            f"logs/{timestr}_gradio_log.log"
        )
        console_handler = logging.StreamHandler()

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        console_handler.setLevel(logging.INFO)
        file_handler.setLevel(logging.INFO)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

        return logger
    
    def generate_prompt_with_history( text, history, kwargs, max_length=2048):
        """ 
        Generate a prompt with history for the TorchON application.
        Args:
            text (str): The text prompt.
            history (list): List of previous conversation messages.
            max_length (int): The maximum length of the prompt.
        Returns:
            tuple: A tuple containing the generated prompt, conversation, and conversation copy. If the prompt could not be generated within the max_length limit, returns None.
        """
        user_role_ind = 0
        bot_role_ind = 1

        # Initialize conversation
        conversation = MyRetriever.retrieve_passages(query=text)

        if history:
            conversation.messages = history

        # REVISE 👇🏻

        conversation = Conversation.append_message(conversation.roles[user_role_ind], text)
        conversation = Conversation.append_message(conversation.roles[bot_role_ind], "")

        # Create a copy of the conversation to avoid history truncation in the UI
        conversation_copy = conversation.copy()
        logger.info("=" * 80)
        logger.info(ChatbotManager.get_prompt(conversation))

        rounds = len(conversation.messages) // 2

        for _ in range(rounds):
            current_prompt = ChatbotManager.get_prompt(conversation)
            if len(conversation.messages) % 2 != 0:
                gr.Error("The messages between user and assistant are not paired.")
                return

            try:
                for _ in range(2):  # pop out two messages in a row
                    conversation.messages.pop(0)
            except IndexError:
                gr.Error("Input text processing failed, unable to respond in this round.")
                return None

        gr.Error("Prompt could not be generated within max_length limit.")
        return None

    def to_gradio_chatbot(conv):
        """Convert the conversation to gradio chatbot format."""
        ret = []
        for i, (role, msg) in enumerate(conv.messages[conv.offset :]):
            if i % 2 == 0:
                ret[-1][-1] = msg
            else:
                ret[-1][-1] = msg
        return ret
    def to_gradio_history(conv):
        """Convert the conversation to gradio history state."""
        return conv.messages[conv.offset :]

    def get_prompt(conv) -> str:
        """Get the prompt for generation."""
        system_prompt = conv.system_template.format(system_message=conv.system_message)
        if conv.sep_style == SeparatorStyle.DeepSeek:
            seps = [conv.sep, conv.sep2]
            if system_prompt == "" or system_prompt is None:
                ret = ""
            else:
                ret = system_prompt + seps[0]
            for i, (role, message) in enumerate(conv.messages):
                if message:
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        else:
            return conv.get_prompt

    def predict(text, chatbot, history, top_p, temperature, repetition_penalty, max_length_tokens, max_context_length_tokens, model_select_dropdown,):
        """
        Function to predict the response based on the user's input and selected model.
        Parameters:
        user_text (str): The input text from the user.
        user_image (str): The input image from the user.
        chatbot (str): The chatbot's name.
        history (str): The history of the chat.
        top_p (float): The top-p parameter for the model.
        temperature (float): The temperature parameter for the model.
        max_length_tokens (int): The maximum length of tokens for the model.
        max_context_length_tokens (int): The maximum length of context tokens for the model.
        model_select_dropdown (str): The selected model from the dropdown.
        Returns:
        generator: A generator that yields the chatbot outputs, history, and status.
        """
        print("running the prediction function")
        conversation = ChatbotManager.generate_prompt_with_history(
            text,
            history,
            max_length=max_context_length_tokens,
        )
        prompts = Conversation.convert_conversation_to_prompts(conversation)
        gradio_chatbot_output = ChatbotManager.to_gradio_chatbot(conversation)

        full_response = ""

        "Generating..."

        print("flushed result to gradio")


        yield gradio_chatbot_output, ChatbotManager.to_gradio_history(conversation), "Generate: Success"


    def retry(
        text,
        chatbot,
        history,
        top_p,
        temperature,
        repetition_penalty,
        max_length_tokens,
        max_context_length_tokens,
        model_select_dropdown,
    ):
        if len(history) == 0:
            yield (chatbot, history, "Empty context")
            return

        chatbot.pop()
        history.pop()
        text = history.pop()[-1]
        # if type(text) is tuple:
        #     text, image = text

        
