import os
import torch
import json # For loading vocabulary

# Assuming data_loader.py and model.py are in the same directory or installable package
try:
    from .model import SimpleLSTMagger
    from .data_loader import tokenize_text # Using tokenize_text from data_loader
except ImportError:
    # Fallback for environments where ComplexTermHighlighter is not a package
    # This might happen if running the script directly without installing the package
    print("Attempting to import SimpleLSTMagger and tokenize_text directly...")
    from model import SimpleLSTMagger
    from data_loader import tokenize_text


# --- Global Variables / Placeholders (should align with training or be loaded) ---
# These values should ideally be loaded from a config file or saved alongside the model.
# For now, they are placeholders and must match the parameters used during training with Russian data.
VOCAB_SIZE = 5000  # Placeholder: Max size of the Russian vocabulary (update if actual is smaller after loading vocab)
EMBEDDING_DIM = 100 # Dimension of token embeddings
HIDDEN_DIM = 128   # Dimension of LSTM hidden states
OUTPUT_DIM = 1     # Output dimension (1 for BCEWithLogitsLoss, 2+ for CrossEntropy)
N_LAYERS = 2       # Number of LSTM layers
DROPOUT = 0.3      # Dropout rate (used during model instantiation)

# Paths to model and vocabulary (relative to the ComplexTermHighlighter directory)
# These should point to the actual saved artifacts from the training process (now for Russian).
DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
MODEL_PATH = os.path.join(DEFAULT_MODEL_DIR, 'term_highlighter_model_russian.pt') # Model trained on Russian data
WORD_TO_IDX_PATH = os.path.join(DEFAULT_MODEL_DIR, 'word_to_idx_russian.json') # Russian vocabulary file

# Special vocabulary tokens (must match those used in training)
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


# --- Utility Functions (similar to those in train.py) ---
def tokens_to_indices(tokens: list[str], word_to_idx: dict) -> list[int]:
    """
    Converts a list of tokens to their corresponding indices.
    Uses UNK_TOKEN for tokens not found in the vocabulary.

    Args:
        tokens (list[str]): The list of tokens to convert.
        word_to_idx (dict): The vocabulary mapping words to indices.

    Returns:
        list[int]: A list of token indices.
    """
    if not word_to_idx:
        raise ValueError("word_to_idx mapping is empty or not loaded.")
    unk_idx = word_to_idx.get(UNK_TOKEN)
    if unk_idx is None:
        # Fallback if UNK_TOKEN itself is not in word_to_idx (should not happen with proper vocab)
        # This indicates a critical issue with vocab setup.
        raise ValueError(f"'{UNK_TOKEN}' not found in word_to_idx. Vocabulary is improperly configured.")
    return [word_to_idx.get(token, unk_idx) for token in tokens]

def pad_sequences(sequences: list[list[int]], batch_first: bool = True, padding_value: int = 0) -> torch.Tensor:
    """
    Pads a list of sequences (lists of token indices) to the same length.
    This version is simplified for handling a single sequence for prediction.

    Args:
        sequences (list[list[int]]): List of sequences (e.g., [[1,2,3]] for a single input).
        batch_first (bool): If True, output tensor will have batch size as the first dimension.
        padding_value (int): The value used for padding.

    Returns:
        torch.Tensor: A PyTorch tensor of the padded sequences.
    """
    if not sequences:
        return torch.empty(0) # Return empty tensor if no sequences
        
    max_len = max(len(seq) for seq in sequences)
    
    padded_sequences = []
    for seq in sequences:
        padded_seq = seq + [padding_value] * (max_len - len(seq))
        padded_sequences.append(padded_seq)
        
    tensor_sequences = torch.tensor(padded_sequences, dtype=torch.long)
    
    if not batch_first:
        tensor_sequences = tensor_sequences.transpose(0, 1)
        
    return tensor_sequences

# --- Core Functions ---
def load_vocabulary(vocab_path: str) -> dict:
    """
    Loads a saved vocabulary (word_to_idx dictionary) from a JSON file.

    Args:
        vocab_path (str): Path to the JSON file containing the word_to_idx map.

    Returns:
        dict: The loaded word_to_idx dictionary.
              Returns an empty dict if the file is not found or there's an error.
    """
    print(f"Loading vocabulary from: {vocab_path}")
    try:
        with open(vocab_path, 'r', encoding='utf-8') as f:
            word_to_idx = json.load(f)
        # Update global VOCAB_SIZE to the actual size of the loaded vocabulary
        # This is important if the placeholder VOCAB_SIZE differs from the trained one.
        global VOCAB_SIZE
        VOCAB_SIZE = len(word_to_idx)
        print(f"Vocabulary loaded successfully. Actual vocab size: {VOCAB_SIZE}")
        return word_to_idx
    except FileNotFoundError:
        print(f"Error: Vocabulary file not found at {vocab_path}.")
        return {}
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from vocabulary file at {vocab_path}.")
        return {}
    except Exception as e:
        print(f"An unexpected error occurred while loading vocabulary: {e}")
        return {}

def load_model(model_path: str, vocab_size: int, embedding_dim: int, 
               hidden_dim: int, output_dim: int, n_layers: int, dropout: float) -> SimpleLSTMagger | None:
    """
    Initializes an instance of SimpleLSTMagger, loads saved model weights,
    and sets the model to evaluation mode.

    Args:
        model_path (str): Path to the saved model state dictionary (.pt file).
        vocab_size (int): Vocabulary size (must match training).
        embedding_dim (int): Embedding dimension (must match training).
        hidden_dim (int): Hidden layer dimension (must match training).
        output_dim (int): Output dimension (must match training).
        n_layers (int): Number of LSTM layers (must match training).
        dropout (float): Dropout rate (must match training).

    Returns:
        SimpleLSTMagger | None: The loaded model, or None if an error occurs.
    """
    print(f"Loading model from: {model_path}")
    try:
        # Instantiate the model with the same architecture as during training
        model = SimpleLSTMagger(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            n_layers=n_layers,
            dropout=dropout  # Dropout is part of the model architecture
        )
        
        # Load the saved state dictionary
        # Ensure the model is loaded to the appropriate device (CPU for inference if GPU not req/avail)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device) # Move model to device
        
        # Set the model to evaluation mode
        # This disables layers like dropout and batch normalization if they were used.
        model.eval()
        print("Model loaded successfully and set to evaluation mode.")
        return model
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}.")
        return None
    except RuntimeError as e:
        print(f"Error loading model state_dict: {e}")
        print("This can happen if model architecture doesn't match the saved weights,")
        print("or if the vocab_size used here differs from training.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while loading the model: {e}")
        return None


def preprocess_input_text(text: str, word_to_idx: dict, tokenizer_name: str = 'nltk') -> tuple[torch.Tensor | None, list[str]]:
    """
    Tokenizes input text, converts to indices, and pads for model input.

    Args:
        text (str): The raw input text string.
        word_to_idx (dict): The loaded vocabulary mapping.
        tokenizer_name (str): The tokenizer to use ('nltk' or future 'transformer').

    Returns:
        tuple[torch.Tensor | None, list[str]]: 
            - A PyTorch tensor of the processed and padded sequence, ready for the model.
              Shape: [1, seq_len] (batch_size of 1). None if preprocessing fails.
            - The list of original tokens.
    """
    if not word_to_idx:
        print("Error: word_to_idx is empty. Cannot preprocess text.")
        return None, []

    print("Preprocessing input text...")
    # 1. Tokenize text
    # Uses tokenize_text from data_loader.py, which has been updated to support Russian
    # (e.g., by passing language='russian' to nltk.word_tokenize).
    # Ensure NLTK's 'punkt' and Russian models are available if using NLTK.
    original_tokens = tokenize_text(text, tokenizer_name=tokenizer_name)
    if not original_tokens:
        print("Warning: Tokenization resulted in no tokens.")
        return None, []
    
    print(f"Tokens: {original_tokens}")

    # 2. Convert tokens to indices
    try:
        indexed_sequence = tokens_to_indices(original_tokens, word_to_idx)
    except ValueError as e:
        print(f"Error converting tokens to indices: {e}")
        return None, original_tokens # Return original tokens for context
    
    print(f"Indexed sequence: {indexed_sequence}")

    # 3. Pad sequence (as a batch of one)
    # The model expects input shape [batch_size, seq_len].
    # Here, batch_size is 1.
    # Padding value should be the index of PAD_TOKEN.
    pad_idx = word_to_idx.get(PAD_TOKEN)
    if pad_idx is None:
        print(f"Error: '{PAD_TOKEN}' not found in word_to_idx. Cannot pad sequence.")
        return None, original_tokens
        
    padded_tensor = pad_sequences([indexed_sequence], batch_first=True, padding_value=pad_idx)
    print(f"Padded tensor shape: {padded_tensor.shape}")
    
    return padded_tensor, original_tokens


def predict_complex_terms(model: SimpleLSTMagger, processed_text_tensor: torch.Tensor) -> list[int]:
    """
    Performs inference on the processed text tensor to predict complex terms.

    Args:
        model (SimpleLSTMagger): The loaded and evaluation-mode PyTorch model.
        processed_text_tensor (torch.Tensor): The input tensor from preprocess_input_text.
                                             Shape: [1, seq_len].

    Returns:
        list[int]: A list of predictions (0 or 1) for each token in the input sequence.
                   Returns an empty list if prediction fails.
    """
    if processed_text_tensor is None or processed_text_tensor.numel() == 0:
        print("Error: Cannot predict with empty or None input tensor.")
        return []

    print("Predicting complex terms...")
    device = next(model.parameters()).device # Get model's device
    processed_text_tensor = processed_text_tensor.to(device)

    try:
        with torch.no_grad(): # Disable gradient calculations for inference
            # outputs shape: [batch_size, seq_len, output_dim]
            # For this model (output_dim=1), shape is [1, seq_len, 1]
            outputs = model(processed_text_tensor)

        # Apply sigmoid and threshold for binary classification (if output_dim=1 and using BCEWithLogitsLoss)
        # outputs.squeeze() removes dimensions of size 1.
        # If batch_size is 1 and output_dim is 1, output shape [1, seq_len, 1] -> squeeze() -> [seq_len]
        # If output_dim > 1 (e.g. for CrossEntropy), use argmax:
        # predictions = torch.argmax(outputs, dim=-1).squeeze().tolist()
        
        # Assuming OUTPUT_DIM = 1 and BCEWithLogitsLoss was used during training:
        probabilities = torch.sigmoid(outputs) # Shape: [1, seq_len, 1]
        # Squeeze to remove batch and output_dim (if 1), then apply threshold
        predictions_tensor = (probabilities.squeeze() > 0.5).int() # Shape: [seq_len]
        
        # Convert tensor to list of integers
        predictions_list = predictions_tensor.tolist()
        
        # Ensure predictions_list is actually a list, not a single int if seq_len was 1
        if not isinstance(predictions_list, list):
            predictions_list = [predictions_list]
            
        print(f"Raw model outputs (first 5): {outputs.squeeze()[:5]}")
        print(f"Probabilities (first 5): {probabilities.squeeze()[:5]}")
        print(f"Predictions (0/1): {predictions_list}")
        return predictions_list

    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return []


def highlight_text(original_tokens: list[str], predictions: list[int], highlight_format: str = "**{term}**") -> str:
    """
    Highlights tokens predicted as complex by wrapping them with a specified format.

    Args:
        original_tokens (list[str]): The list of original tokens from tokenization.
        predictions (list[int]): List of 0s or 1s, corresponding to original_tokens.
        highlight_format (str): A format string for highlighting, e.g., "**{term}**"
                                or "<span class='complex'>{term}</span>".

    Returns:
        str: A string with complex terms highlighted.
    """
    if len(original_tokens) != len(predictions):
        print("Warning: Mismatch between token count and prediction count. Cannot highlight reliably.")
        # Fallback: join original tokens without highlighting
        return " ".join(original_tokens)

    highlighted_tokens = []
    for token, pred in zip(original_tokens, predictions):
        if pred == 1:
            highlighted_tokens.append(highlight_format.format(term=token))
        else:
            highlighted_tokens.append(token)
    
    # Simple join with space. For more accurate reconstruction, consider original spacing.
    return " ".join(highlighted_tokens)


# --- Main Orchestration Function ---
def get_highlighted_text(text_input: str, model_path: str, vocab_path: str) -> str:
    """
    Orchestrates loading the model, preprocessing text, predicting, and highlighting.
    # This highlighter is intended for use with models trained on Russian text and a Russian vocabulary.

    Args:
        text_input (str): The raw text string to process (assumed to be Russian).
        model_path (str): Path to the saved PyTorch model file (trained on Russian data).
        vocab_path (str): Path to the saved vocabulary JSON file (for Russian).

    Returns:
        str: The text with complex terms highlighted, or the original text if errors occur.
    """
    print("\n--- Starting Highlighting Process ---")
    
    # 1. Load Vocabulary
    word_to_idx = load_vocabulary(vocab_path)
    if not word_to_idx:
        print("Failed to load vocabulary. Cannot proceed.")
        return f"[Error: Vocabulary not loaded] {text_input}"
    
    # Check if PAD_TOKEN is in vocab, critical for padding.
    if PAD_TOKEN not in word_to_idx:
        print(f"Critical Error: '{PAD_TOKEN}' is not in the loaded vocabulary. Highlighting may fail.")
        # Depending on strictness, one might return or try to proceed with caution.
        # return f"[Error: Invalid Vocabulary ({PAD_TOKEN} missing)] {text_input}"

    # Actual vocab size from loaded vocab should be used for model loading
    loaded_vocab_size = len(word_to_idx)

    # 2. Load Model
    # Uses global placeholders for model architecture params, ideally these should be saved with model.
    model = load_model(
        model_path=model_path, 
        vocab_size=loaded_vocab_size, # Use actual loaded vocab size
        embedding_dim=EMBEDDING_DIM, 
        hidden_dim=HIDDEN_DIM, 
        output_dim=OUTPUT_DIM, 
        n_layers=N_LAYERS, 
        dropout=DROPOUT
    )
    if model is None:
        print("Failed to load model. Cannot proceed.")
        return f"[Error: Model not loaded] {text_input}"

    # 3. Preprocess Input Text
    processed_tensor, original_tokens = preprocess_input_text(text_input, word_to_idx)
    if processed_tensor is None or not original_tokens:
        print("Failed to preprocess text. Cannot proceed with highlighting.")
        return f"[Error: Preprocessing failed] {text_input}"

    # 4. Get Predictions
    predictions = predict_complex_terms(model, processed_tensor)
    if not predictions and original_tokens: # If prediction failed but we have tokens
        print("Failed to get predictions. Returning original text.")
        return " ".join(original_tokens) # Or just text_input
    if len(predictions) != len(original_tokens):
        print(f"Warning: Number of predictions ({len(predictions)}) does not match number of tokens ({len(original_tokens)}).")
        # Fallback to original text or try to make sense of it based on requirements.
        # For safety, returning original text or a message.
        return f"[Error: Prediction length mismatch] {' '.join(original_tokens)}"


    # 5. Highlight Text
    highlighted_output = highlight_text(original_tokens, predictions)
    print("\n--- Highlighting Process Finished ---")
    return highlighted_output


# --- Entry Point for Example Usage ---
if __name__ == '__main__':
    print("Highlighter script running in __main__ block (for demonstration).")
    print("This example assumes a trained model and vocabulary exist at specified paths.")
    print(f"Expected model path: {MODEL_PATH}")
    print(f"Expected vocabulary path: {WORD_TO_IDX_PATH}")

    # --- Prerequisite Check ---
    # Check if NLTK 'punkt' is available (needed by data_loader.tokenize_text which uses NLTK).
    # For Russian, data_loader.py also specifies language='russian' in nltk.word_tokenize.
    try:
        import nltk
        # Test if 'punkt' and Russian tokenization capabilities are available
        nltk.word_tokenize("тест", language='russian') 
    except LookupError:
        print("\nNLTK 'punkt' resource (and potentially Russian models) not found. Attempting to download...")
        try:
            nltk.download('punkt', quiet=False) 
            # nltk.download('russian') # May also be needed
            print("'punkt' (and potentially other Russian NLTK resources) downloaded successfully.")
            nltk.word_tokenize("тест", language='russian') # Verify after download
        except Exception as e:
            print(f"Failed to download NLTK resources automatically: {e}")
            print("Please ensure 'punkt' (and 'russian' models if needed) are available for NLTK.")
    except ImportError:
        print("NLTK library is not installed. Please install it: pip install nltk")


    # --- Create Dummy Model and Vocab for Demo if they don't exist ---
    # This is purely for allowing the __main__ block to run without a pre-trained model.
    # In a real scenario, these files MUST be products of the training script (for Russian).
    
    # Ensure 'models' directory exists
    if not os.path.exists(DEFAULT_MODEL_DIR):
        print(f"Creating dummy models directory: {DEFAULT_MODEL_DIR}")
        os.makedirs(DEFAULT_MODEL_DIR)

    # Dummy Vocabulary (word_to_idx_russian.json)
    # MODEL_PATH and WORD_TO_IDX_PATH globals are already updated to _russian versions
    if not os.path.exists(WORD_TO_IDX_PATH):
        print(f"Creating dummy Russian vocabulary file at: {WORD_TO_IDX_PATH}")
        dummy_vocab_russian = {
            PAD_TOKEN: 0, UNK_TOKEN: 1, "это": 2, "простое": 3, "предложение": 4, 
            "для": 5, "тестирования": 6, "с": 7, "некоторыми": 8, "потенциально": 9, 
            "сложными": 10, "терминами": 11, "и": 12, "жаргоном": 13, ".": 14
            # This is a tiny example, a real Russian vocab would be much larger.
        }
        VOCAB_SIZE = len(dummy_vocab_russian) 
        with open(WORD_TO_IDX_PATH, 'w', encoding='utf-8') as f_vocab:
            json.dump(dummy_vocab_russian, f_vocab, ensure_ascii=False) # ensure_ascii=False for Cyrillic
    else:
        print(f"Using existing vocabulary file: {WORD_TO_IDX_PATH}")
        temp_vocab = load_vocabulary(WORD_TO_IDX_PATH) # load_vocabulary updates global VOCAB_SIZE
        if not temp_vocab:
             print("Warning: Could not load existing vocab to set VOCAB_SIZE for dummy model creation.")


    # Dummy Model (term_highlighter_model_russian.pt)
    if not os.path.exists(MODEL_PATH):
        print(f"Creating and saving a dummy model for Russian at: {MODEL_PATH}")
        print(f"Dummy model will use VOCAB_SIZE = {VOCAB_SIZE} based on existing or new dummy Russian vocab.")
        dummy_model = SimpleLSTMagger(
            vocab_size=VOCAB_SIZE, 
            embedding_dim=EMBEDDING_DIM, 
            hidden_dim=HIDDEN_DIM, 
            output_dim=OUTPUT_DIM, 
            n_layers=1, 
            dropout=0.0
        )
        torch.save(dummy_model.state_dict(), MODEL_PATH)
        print("Dummy Russian model created. Its predictions will be random.")
    else:
        print(f"Using existing model file: {MODEL_PATH}")
        print("Ensure this model was trained with compatible parameters for Russian data.")

    # --- Example Usage (with Russian text) ---
    sample_text_to_highlight = "Это простое предложение для тестирования с некоторыми потенциально сложными терминами и жаргоном."
    print(f"\nInput Russian Text: '{sample_text_to_highlight}'")

    highlighted_result = get_highlighted_text(
        text_input=sample_text_to_highlight,
        model_path=MODEL_PATH, 
        vocab_path=WORD_TO_IDX_PATH
    )

    print(f"\nHighlighted Russian Text: {highlighted_result}")

    print("\n--- Note on Dummy Artifacts ---")
    print("If dummy model/vocabulary were created, highlighting is based on random weights or a tiny Russian vocab.")
    print(f"For meaningful results, ensure '{MODEL_PATH}' and '{WORD_TO_IDX_PATH}' are actual trained artifacts from 'train.py' using Russian data.")
    print("The script attempts to use the actual size of the loaded vocabulary for the model.")
