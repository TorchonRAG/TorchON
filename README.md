## 🔦TorchON : Adaptable Retrieval Augmented Generation

> it's only real torchon if it comes from the Rague region of France


🔦TorchON is an adaptable retrieval augmented application that provides question answering over documents, GitHub repositories, and websites. It takes data, creates synthetic data, and uses that synthetic data to optimize the prompts of the 🔦Torchon application. The application recompiles itself every run in a unique and adapted way to the user query. You can then decide to publish this application locally, privately, or share it with selected people.

### Table of Contents

- [Introduction](#introduction)
- [Setup](#setup)
- [How It Works](#how-it-works)
- [Contributing](#contributing)
- [License](#license)

### Local Models & Enterprise Security

🔦TorchON is an innovative application that leverages the power of retrieval augmented generation to provide accurate and relevant answers to user queries. By adapting itself to each query, 🔦TorchON ensures that the generated responses are tailored to the specific needs of the user.

- 🔦TorchON uses local models for embeddings : you never send your data over the internet
- 🔦TorchON uses Anthropic/Claude-3-Opus : your querries are not used for training

### Setup

To set up 🔦TorchON, follow these steps:

1. Clone the repository:
```bash
git clone https://github.com/Tonic-AI/torchon.git
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```
3. Run the application:
```bash
python app.py
```
4. Follow the instructions on screen to add your own access keys, files, and fine tune your application.

### How It Works

TorchON works by following these key steps:

1. **Data Collection**: The application collects data from various sources, including documents, GitHub repositories, and websites. It utilizes different reader classes such as `CSVReader`, `DocxReader`, `PDFReader`, `ChromaReader`, and `SimpleWebPageReader` to extract information from these sources.

2. **Synthetic Data Generation**: TorchON generates synthetic data using the collected data. It employs techniques such as data augmentation and synthesis to create additional training examples that can help improve the performance of the application.

3. **Prompt Optimization**: The synthetic data is used to optimize the prompts of the TorchON application. By fine-tuning the prompts based on the generated data, the application can generate more accurate and relevant responses to user queries.

4. **Recompilation**: Torch recompiles itself every run based on the optimized prompts and the specific user query. This dynamic recompilation allows the application to adapt and provide tailored responses to each query.

5. **Question Answering**: Once recompiled, TorchON takes the user query and retrieves relevant information from the collected data sources. It then generates a response using the optimized prompts and the retrieved information, providing accurate and contextually relevant answers to the user.

6. **Secure & Shareable**: You love the app you made and want to keep it or share it with your folks ? simply configure your access keys to any of our supported cloud service provider to securely share your app in your private networks.

### Contributing

We welcome contributions to 🔦TorchON! If you'd like to contribute, please follow these steps:

1. Fork the repository from GitLab.
```bash
git clone https://git.tonic-ai.com/Tonic-AI/TorchON/torchon.git
```

2. Create a new branch from the `devbranch`:
```bash
git checkout -b feature/your-feature-name devbranch
```

3. Make your changes and commit them with descriptive commit messages.

4. Push your changes to your forked repository:
```bash
git push origin feature/your-feature-name
```

5. Open a pull request against the `devbranch` of the gitlab repository.

Please ensure that your contributions adhere to the project's coding conventions and include appropriate tests and documentation.

## License

TorchON is released under the XXXX License. See the [LICENSE](LICENSE) file for more details.