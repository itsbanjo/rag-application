# Generic RAG Chatbot Application

![RAG Chatbot Screenshot](artefacts/screenshot.png)

## Disclosure Statement

**IMPORTANT: Please read this disclosure statement carefully before using this application.**

This RAG (Retrieval-Augmented Generation) Chatbot application is provided strictly for educational, proof-of-concept, and exploratory purposes only. It is not intended for production use, commercial deployment, or any mission-critical applications.

- This application is an experimental project and may contain errors, bugs, or security vulnerabilities.
- The responses generated by this chatbot are based on machine learning models and may not always be accurate, complete, or appropriate for all situations.
- The information retrieved and presented by this application should not be considered as professional advice (medical, legal, financial, or otherwise).
- Users are solely responsible for verifying any information obtained through this application from authoritative sources before making any decisions based on such information.

Neither the author(s) of this application, Elastic, nor any affiliated parties assume any responsibility or liability for any direct, indirect, incidental, consequential, special, or exemplary damages or losses arising from the use or misuse of this application or the information it provides.

By using this application, you acknowledge that you have read and understood this disclosure statement and agree to use the application at your own risk.

It is strongly recommended to review and comply with all applicable laws, regulations, and ethical guidelines when using and adapting this application for any purpose.

---

This is a Retrieval-Augmented Generation (RAG) Chatbot application that uses Elasticsearch for information retrieval and OpenAI's GPT model for generating responses. The application is built with Streamlit for a user-friendly interface.

## Prerequisites

- Python 3.7+
- An Elasticsearch cluster
- An OpenAI API key

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/itsbanjo/rag-application.git
   cd rag-application
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

For macOS users, please refer to the [macOS Python Installation Guide](#macos-python-installation-guide) below.

## Configuration

Create a `.env` file in the root directory of the project based on the provided `.env.example` file. Here's an explanation of each environment variable:

- `CLOUD_ID`: Your Elasticsearch Cloud ID
- `API_KEY`: Your Elasticsearch API key
- `ELASTIC_URL`: The URL of your Elasticsearch instance
- `LLM_TYPE`: The type of language model to use (currently only supports 'openai')
- `OPENAI_API_KEY`: Your OpenAI API key
- `INDEX_NAME`: The name of the Elasticsearch index to search
- `TITLE`: The preferred named of the application or the project
- `LOGO_PATH`: Path to the logo file for the application (optional)
- `PROMPT_TEMPLATE_FILE`: Path to the prompt template file (optional)
- `ELSER_MODEL`: The name of the ELSER model to use for Elasticsearch (default: ".elser_model_2_linux-x86_64")

Example `.env` file:

```
CLOUD_ID=your-cloud-id
API_KEY=your-api-key
ELASTIC_URL=https://your-elasticsearch-url
LLM_TYPE=openai
OPENAI_API_KEY=your-openai-api-key
INDEX_NAME=your-index-name
LOGO_PATH=artefacts/your-logo.png
PROMPT_TEMPLATE_FILE=artefacts/prompt_template.txt
ELSER_MODEL=".elser_model_2_linux-x86_64"
```

## Running the Application

To run the application, use the following command:

```
streamlit run rag-application.py
```

For debugging purposes, you can enable debug mode by setting the `DEBUG` environment variable:

```
DEBUG=true streamlit run rag-application.py
```

This will provide additional logging information in the console and the Streamlit interface.

## Usage

1. Once the application is running, open your web browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).
2. Enter your question in the input field.
3. The application will retrieve relevant information from Elasticsearch and generate a response using the OpenAI model.
4. The response will be displayed along with cited references.

## Customization

- To customize the prompt template, edit the file specified in `PROMPT_TEMPLATE_FILE`.
- To change the application logo, update the `LOGO_PATH` in the `.env` file.

## Troubleshooting

If you encounter any issues:

1. Check that all environment variables are correctly set in your `.env` file.
2. Ensure you have the necessary permissions for your Elasticsearch cluster and OpenAI API key.
3. Run the application in debug mode to get more detailed logging information.

## macOS Python Installation Guide

### 1. Install Homebrew (if not already installed)

Homebrew is a package manager for macOS that makes it easy to install software. Open Terminal and run:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Follow the prompts to complete the installation.

### 2. Install Python

We'll use Homebrew to install Python. In Terminal, run:

```bash
brew install python
```

This will install the latest version of Python 3.

### 3. Verify Python Installation

Check that Python is installed correctly:

```bash
python3 --version
```

This should display the installed Python version.

### 4. Set up a Virtual Environment

#### a. Navigate to Your Project Directory

In Terminal, navigate to your project directory:

```bash
cd path/to/your/project
```

#### b. Create a Virtual Environment

Create a new virtual environment in your project directory:

```bash
python3 -m venv venv
```

#### c. Activate the Virtual Environment

To activate the virtual environment, run:

```bash
source venv/bin/activate
```

### 5. Install Required Packages

With your virtual environment activated, install the required packages:

```bash
pip install -r requirements.txt
```

### 6. Deactivating the Virtual Environment

When you're done, you can deactivate the virtual environment:

```bash
deactivate
```

### 7. Using the Virtual Environment

Every time you want to work on your project, activate the virtual environment:

```bash
cd path/to/your/project
source venv/bin/activate
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
