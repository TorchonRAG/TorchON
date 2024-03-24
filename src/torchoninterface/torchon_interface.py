import gradio as gr
from typing import Any
from main import Application
from src.torchonragmaker.torchonragmaker import CreateRAG

class TorchonInterface:
    def __init__(self, application):
        self.application = Application(self)

    def launch_interface(self):
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
                        fn=self.application.set_api_keys,
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
                        fn=self.application.handle_file_upload,
                        inputs=[file_upload, folder_upload, webpage_input, github_input],
                        outputs=validation_output
                    )

                with gr.Accordion("Generate Synthetic Data") as generate_data_accordion:
                    schema_input = gr.Textbox(label="Schema Class Name")
                    sample_size_input = gr.Number(label="Sample Size", value=100)
                    synthetic_data_button = gr.Button("Generate Synthetic Data")
                    synthetic_data_output = gr.Textbox()

                    synthetic_data_button.click(
                        fn=self.application.handle_synthetic_data,
                        inputs=[schema_input, sample_size_input],
                        outputs=synthetic_data_output
                    )

    #             with gr.Accordion("Chatbot") as chatbot_accordion:
    #                 text_input = gr.Textbox(label="Enter your question")
    # #               model_select = gr.Dropdown(label="Select Model", choices=list(self.chatbot_manager.models.keys()))
    #                 top_p_input = gr.Slider(label="Top-p", min_value=0.0, max_value=1.0, value=0.95, step=0.01)
    #                 temperature_input = gr.Slider(label="Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.01)
    #                 repetition_penalty_input = gr.Slider(label="Repetition Penalty", min_value=1.0, max_value=2.0, value=1.1, step=0.1)
    #                 max_length_tokens_input = gr.Number(label="Max Length Tokens", value=2048)
    #                 max_context_length_tokens_input = gr.Number(label="Max Context Length Tokens", value=2048)
    #                 chatbot_output = gr.Chatbot(label="Chatbot Conversation")
    #                 submit_button = gr.Button("Submit")

    #                 submit_button.click(
    #                     fn=self.handle_chatbot_interaction, 
    #                     inputs=[text_input, model_select,
    #                             top_p_input, temperature_input,
    #                             repetition_penalty_input,
    #                             max_length_tokens_input,
    #                             max_context_length_tokens_input
    #                         ],    outputs=chatbot_output
    #                     )

    #             with gr.Accordion("Publish"):
    #                 title_input = gr.Textbox(label="Title")
    #                 system_prompt_input = gr.Textbox(label="System prompt", lines=3)
    #                 example_input = gr.Textbox(label="Example", lines=2)
    #                 hf_token_input = gr.Textbox(label="Hugging Face Token", type="password")
    #                 publish_button = gr.Button("Publish")
                    
    #                 def update_publish_info(title, system_prompt, example, hf_token):
    #                     self.title = title
    #                     self.system_prompt = system_prompt
    #                     self.example = example
    #                     self.hf_token = hf_token
    #                     return self.publish()
                    
    #                 publish_button.click(
    #                     fn=update_publish_info,
    #                     inputs=[title_input, system_prompt_input, example_input, hf_token_input],
    #                     outputs=[]
    #                 )

                with gr.Accordion("SimpleTestingChatbot"):
                    with gr.Tab("Create"):
                        chatbot_maker = gr.Chatbot(layout="panel", elem_id="chatbot-maker")
                        textbox_maker = gr.Textbox(placeholder="Your words here", autofocus=True)
                        submit_btn = gr.Button("Send")

                    submit_btn.click(
                        fn=self.application.CreateRAG,
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
                            fn=self.application.publish,
                            inputs=[title_input, system_prompt_input, example_input, hf_token_input],
                            outputs=[]
                        )

            demo.launch()


    def launch(application: Any):
        interface = TorchonInterface(application)
        interface.launch_interface()
