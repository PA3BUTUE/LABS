# Russian Complex Term Highlighter

## Description

This project aims to identify and highlight complex terms within Russian texts. It utilizes a PyTorch-based model for the underlying sequence tagging task. This is currently a foundational implementation demonstrating the core components of such a system, with a focus on Russian language processing.

## Features (Planned/Implemented)

*   **Text Loading**: Supports loading text from `.txt` files.
*   **Tokenization**:
    *   Implemented: NLTK-based tokenization (`nltk.word_tokenize` with `language='russian'`).
    *   Planned/Considered: Integration of more advanced Russian-specific tokenizers like `pymorphy2` for morphological analysis or tokenizers from Hugging Face Transformers (e.g., for BERT-like models).
*   **Complex Term Identification**:
    *   Implemented: Currently uses exact list matching against a predefined list of complex terms.
    *   Planned: Enhancements to use machine learning-based classification for more dynamic and context-aware term identification.
*   **Term Highlighting**: Highlights identified complex terms in the output text (e.g., using Markdown `**term**`).
*   **Command-Line Interface**: `main.py` provides a CLI for:
    *   Training a new model.
    *   Highlighting terms in a text file using a pre-trained model.

## Directory Structure

*   `data/`: Contains sample data for training and testing (e.g., `sample.txt`, `complex_terms.txt`).
*   `models/`: Intended directory for saving trained model artifacts (e.g., `term_highlighter_model_russian.pt`, `word_to_idx_russian.json`). *Note: This directory might be created by the training script if it doesn't exist.*
*   `data_loader.py`: Handles loading text data, tokenization, and labeling tokens based on complex term lists. Includes considerations for Russian text.
*   `model.py`: Defines the PyTorch neural network model (a simple LSTM-based tagger). Includes comments on adapting for Russian embeddings.
*   `train.py`: Orchestrates the model training process, including vocabulary building, data splitting, training loop, and model saving. Adapted for Russian data paths.
*   `highlighter.py`: Loads a pre-trained model and its vocabulary to perform inference and highlight complex terms in new text. Adapted for Russian model paths.
*   `main.py`: Provides the command-line interface for interacting with the project (training and highlighting).
*   `requirements.txt`: Lists the Python dependencies for the project.
*   `README.md`: This file, providing an overview and instructions for the project.

## Setup and Installation

1.  **Python Version**: Python 3.8+ is recommended.
2.  **Dependencies**: The project dependencies are listed in `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```
3.  **Crucial Environment Note**:
    *   **Note: The current development environment has disk space limitations that prevent the installation of heavy dependencies like PyTorch and Transformers. Full functionality, including training and running the model, requires an environment where these libraries can be installed.**
4.  **Ideal Dependencies for Russian NLP (as in `requirements.txt`)**:
    *   `torch`: For PyTorch model development and execution.
    *   `transformers`: For using pre-trained transformer models and tokenizers (especially useful for Russian).
    *   `nltk`: For basic NLP tasks like tokenization.
    *   `scikit-learn`: For utilities like data splitting.
    *   `pymorphy2`: (Optional, but recommended for advanced Russian morphology analysis and lemmatization).
5.  **NLTK Data for Russian**:
    If using NLTK for tokenization, you may need to download specific resources:
    ```python
    import nltk
    nltk.download('punkt')  # For tokenization
    nltk.download('russian') # For Russian-specific models, if NLTK's tokenizer uses them beyond 'punkt' for Russian.
    ```
    The scripts (`train.py`, `highlighter.py`) attempt to guide on this if resources are missing.

## Usage

The primary interface to the project is `main.py`.

### Training a New Model

To train a new model (assuming you have appropriate Russian training data in `data/` and an environment where training is feasible):

```bash
python ComplexTermHighlighter/main.py train --text_file ComplexTermHighlighter/data/sample.txt --complex_terms_file ComplexTermHighlighter/data/complex_terms.txt --model_save_dir ComplexTermHighlighter/models/
```
*   `--text_file`: Path to your Russian training text.
*   `--complex_terms_file`: Path to your list of complex Russian terms.
*   `--model_save_dir`: Directory where the trained model (e.g., `term_highlighter_model_russian.pt`) and vocabulary (e.g., `word_to_idx_russian.json`) will be saved.

*Note: The `train.py` script currently uses some hardcoded hyperparameters and paths internally. For full CLI control, `train.py` would require refactoring to accept all relevant parameters.*

### Highlighting Complex Terms

To highlight complex terms in a Russian text file using a pre-trained model:

```bash
python ComplexTermHighlighter/main.py highlight --input_file ComplexTermHighlighter/data/sample.txt --model_path ComplexTermHighlighter/models/term_highlighter_model_russian.pt --vocab_path ComplexTermHighlighter/models/word_to_idx_russian.json
```
*   `--input_file`: Path to the Russian `.txt` file you want to process.
*   `--model_path`: Path to the pre-trained Russian model file (e.g., `term_highlighter_model_russian.pt`).
*   `--vocab_path`: Path to the corresponding Russian vocabulary file (e.g., `word_to_idx_russian.json`).
*   `--output_file` (Optional): Path to save the highlighted text. If not provided, output is printed to the console.

*Note: The paths to `model_path` and `vocab_path` are illustrative. You must have a trained model and its vocabulary available at these locations for highlighting to work. The example paths point to where a model trained for Russian might be saved.*

## Future Improvements

*   **Advanced Term Identification**: Implement sequence tagging models (e.g., fine-tuning BERT or other transformer-based models pre-trained on Russian corpora like `DeepPavlov/rubert-base-cased`) for more accurate and context-aware complex term identification.
*   **Russian Morphology Handling**: Integrate robust lemmatization (e.g., using `pymorphy2`) during preprocessing for both training data and complex term lists to better handle Russia's rich morphology.
*   **Improved Evaluation Metrics**: Add more sophisticated evaluation for the term identification task (Precision, Recall, F1-score).
*   **User Interface**: Develop a simple web interface or desktop application for easier interaction.
*   **Configurable Parameters**: Refactor `train.py` to fully accept hyperparameters and paths via CLI arguments.
*   **Packaging**: Package the project for easier distribution and installation.
