import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from collections import Counter

# Assuming data_loader.py and model.py are in the same directory
from .data_loader import preprocess_data
from .model import SimpleLSTMagger

# --- Hyperparameters and Configuration ---
# These would typically be loaded from a config file or command-line arguments
VOCAB_SIZE = 5000  # Max size of the vocabulary. # Adjust based on the target Russian vocabulary size. Russian typically has a larger vocabulary due to rich morphology.
EMBEDDING_DIM = 100 # Dimension of token embeddings
HIDDEN_DIM = 128   # Dimension of LSTM hidden states
OUTPUT_DIM = 1     # Output dimension. 1 for BCEWithLogitsLoss (binary complex/not-complex)
                   # Set to 2+ for CrossEntropyLoss if outputting class scores.
N_LAYERS = 2       # Number of LSTM layers
DROPOUT = 0.3      # Dropout rate
LEARNING_RATE = 0.001 # Learning rate for the optimizer
EPOCHS = 10        # Number of training epochs
BATCH_SIZE = 32    # Batch size for training

# Paths (ensure these directories/files exist or are created)
# Example: Create a 'data' subdirectory in ComplexTermHighlighter for these files
# and a 'models' subdirectory for saving the trained model.
# These paths should point to your training data, which is now assumed to be in Russian (e.g., the sample_text.txt and complex_terms.txt files were updated with Russian content).
TEXT_FILE_PATH = 'ComplexTermHighlighter/data/sample.txt' # Path to the input text file (now Russian)
COMPLEX_TERMS_FILE_PATH = 'ComplexTermHighlighter/data/complex_terms.txt' # Path to complex terms list (now Russian)
MODEL_SAVE_PATH = 'ComplexTermHighlighter/models/term_highlighter_model_russian.pt' # Path to save the trained model (updated for Russian)

# Special vocabulary tokens
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

# --- Vocabulary Building ---
def build_vocab(all_tokens_list: list[list[str]], max_vocab_size: int) -> (dict, dict):
    """
    Builds a vocabulary from a list of token lists.

    Args:
        all_tokens_list (list[list[str]]): A list where each element is a list of tokens from a document/sentence.
        max_vocab_size (int): The maximum size of the vocabulary.

    Returns:
        tuple: (word_to_idx, idx_to_word) dictionaries.
    """
    # For Russian, ensure proper handling of Cyrillic characters and potentially use a lemmatizer (e.g., from pymorphy2)
    # on tokens before building vocab to reduce vocabulary size and group inflected forms.
    print("Building vocabulary...")
    all_tokens_flat = [token for sublist in all_tokens_list for token in sublist]
    
    token_counts = Counter(all_tokens_flat)
    most_common_tokens = token_counts.most_common(max_vocab_size - 2) # -2 for PAD and UNK

    word_to_idx = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    idx_to_word = {0: PAD_TOKEN, 1: UNK_TOKEN}

    for i, (token, _) in enumerate(most_common_tokens):
        word_to_idx[token] = i + 2 # Start indexing from 2
        idx_to_word[i + 2] = token
    
    actual_vocab_size = len(word_to_idx)
    print(f"Vocabulary built. Actual size: {actual_vocab_size} (Max requested: {max_vocab_size})")
    if actual_vocab_size < VOCAB_SIZE: # Update global VOCAB_SIZE if smaller
        global VOCAB_SIZE
        VOCAB_SIZE = actual_vocab_size
        print(f"Global VOCAB_SIZE updated to actual: {VOCAB_SIZE}")

    return word_to_idx, idx_to_word

# --- Token to Indices Conversion ---
def tokens_to_indices(tokens: list[str], word_to_idx: dict) -> list[int]:
    """
    Converts a list of tokens to their corresponding indices using word_to_idx.

    Args:
        tokens (list[str]): The list of tokens to convert.
        word_to_idx (dict): The vocabulary mapping words to indices.

    Returns:
        list[int]: A list of token indices.
    """
    return [word_to_idx.get(token, word_to_idx[UNK_TOKEN]) for token in tokens]

# --- Padding Function ---
def pad_sequences(sequences: list[list[int]], batch_first: bool = True, padding_value: int = 0) -> torch.Tensor:
    """
    Pads a list of sequences (lists of token indices) to the same length.

    Args:
        sequences (list[list[int]]): List of sequences (each sequence is a list of integer token indices).
        batch_first (bool): If True, output tensor will have batch size as the first dimension.
        padding_value (int): The value used for padding.

    Returns:
        torch.Tensor: A PyTorch tensor of the padded sequences.
    """
    # Find the maximum length of any sequence in the batch
    max_len = max(len(seq) for seq in sequences) if sequences else 0
    
    padded_sequences = []
    for seq in sequences:
        # Create a new list by appending padding_value until it reaches max_len
        padded_seq = seq + [padding_value] * (max_len - len(seq))
        padded_sequences.append(padded_seq)
        
    tensor_sequences = torch.tensor(padded_sequences, dtype=torch.long)
    
    if not batch_first:
        # Transpose to [seq_len, batch_size]
        tensor_sequences = tensor_sequences.transpose(0, 1)
        
    return tensor_sequences

# --- Main Training Function ---
def train_model():
    """
    Orchestrates the model training process.
    """
    print("Starting model training process...")

    # 1. Load and Preprocess Data
    # For this example, we assume preprocess_data returns a list of token lists and a list of label lists.
    # In a real scenario, you might load multiple files or a dataset object.
    print(f"Loading data from: {TEXT_FILE_PATH} and {COMPLEX_TERMS_FILE_PATH}")
    # Note: preprocess_data is expected to return: all_tokens (list of lists), all_labels (list of lists)
    # This structure needs to be consistent with how build_vocab and subsequent steps expect data.
    # For simplicity, let's assume preprocess_data gives us a flat list of tokens for the entire dataset
    # and corresponding flat list of labels. We'll then need to handle splitting and batching.
    # However, the prompt implies preprocess_data might give document-level token/label lists.
    # Let's adjust to what `preprocess_data` likely provides (tokens, labels for a single text file).
    
    # For the purpose of this script, let's assume `preprocess_data` handles one document at a time.
    # A more robust pipeline would handle multiple documents, aggregate them, then build vocab.
    # Here, we'll simulate having a list of token sequences and label sequences.
    # This is a simplification. Typically, you'd process a whole dataset.
    tokens, labels = preprocess_data(TEXT_FILE_PATH, COMPLEX_TERMS_FILE_PATH, tokenizer_name='nltk')
    
    if not tokens:
        print("No tokens were loaded. Exiting training.")
        return

    # For demonstration, we'll treat the single loaded document's tokens/labels as our "dataset".
    # In a real application, `all_token_sequences` would be a list of many such `tokens` lists.
    all_token_sequences = [tokens] # A list containing one sequence of tokens
    all_label_sequences = [labels] # A list containing one sequence of labels

    # 2. Build Vocabulary
    # The global VOCAB_SIZE might be updated by build_vocab if the corpus is small
    word_to_idx, idx_to_word = build_vocab(all_token_sequences, VOCAB_SIZE)
    
    # 3. Convert all token sequences to index sequences
    indexed_sequences = [tokens_to_indices(seq, word_to_idx) for seq in all_token_sequences]
    # `all_label_sequences` are already numerical (0 or 1)

    # 4. Split Data (Example: 80% train, 20% validation)
    # This split is very basic as we only have one "document" in this example.
    # In a real scenario, you'd have many documents to split.
    if len(indexed_sequences) < 2 and len(indexed_sequences) > 0 : # Need at least 2 samples for train_test_split
        print("Warning: Only one data sample available. Using it for both training and validation (not recommended).")
        train_sequences, val_sequences = indexed_sequences, indexed_sequences
        train_labels, val_labels = all_label_sequences, all_label_sequences
    elif not indexed_sequences:
        print("No data to train on. Exiting.")
        return
    else:
        train_sequences, val_sequences, train_labels, val_labels = train_test_split(
            indexed_sequences, all_label_sequences, test_size=0.2, random_state=42
        )
    
    print(f"Training samples: {len(train_sequences)}, Validation samples: {len(val_sequences)}")

    # 5. Initialize Model
    # For Russian text, ensure VOCAB_SIZE and other parameters are appropriate.
    # Consider using a model architecture or embeddings pre-trained on Russian data if possible.
    print("Initializing model...")
    model = SimpleLSTMagger(
        vocab_size=VOCAB_SIZE, # Use the potentially updated VOCAB_SIZE
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=OUTPUT_DIM,
        n_layers=N_LAYERS,
        dropout=DROPOUT
    )
    print(model)

    # 6. Define Loss Function and Optimizer
    # Using BCEWithLogitsLoss because OUTPUT_DIM = 1 (expects raw logits from model)
    # This loss combines a Sigmoid layer and the BCELoss in one single class.
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # --- Device Configuration (GPU if available) ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)
    criterion.to(device)

    # --- Training Loop ---
    print("\n--- Starting Training ---")
    for epoch in range(EPOCHS):
        model.train() # Set model to training mode
        epoch_loss = 0
        num_batches = (len(train_sequences) + BATCH_SIZE - 1) // BATCH_SIZE # Calculate number of batches

        for i in range(num_batches):
            # Get batch
            start_idx = i * BATCH_SIZE
            end_idx = start_idx + BATCH_SIZE
            batch_sequences = train_sequences[start_idx:end_idx]
            batch_labels_list = train_labels[start_idx:end_idx]

            if not batch_sequences: # Skip if batch is empty
                continue

            # Pad sequences and labels for the current batch
            # Input sequences are token indices
            padded_input_batch = pad_sequences(batch_sequences, batch_first=True, padding_value=word_to_idx[PAD_TOKEN])
            
            # Labels need to be padded to the same sequence length as inputs
            # Their padding value doesn't matter as much if loss function ignores padding,
            # but for consistency, use 0 or a specific ignore_index if applicable.
            # For BCEWithLogitsLoss, labels should be float.
            max_len_batch = padded_input_batch.shape[1]
            padded_labels_batch_list = []
            for lbl_seq in batch_labels_list:
                padded_lbl_seq = lbl_seq + [0] * (max_len_batch - len(lbl_seq)) # Pad with 0
                padded_labels_batch_list.append(padded_lbl_seq)
            
            labels_batch_tensor = torch.tensor(padded_labels_batch_list, dtype=torch.float).to(device)
            input_batch_tensor = padded_input_batch.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            # outputs shape: [batch_size, seq_len, output_dim]
            outputs = model(input_batch_tensor)
            
            # Calculate loss
            # For BCEWithLogitsLoss, outputs are logits, labels are 0 or 1.
            # Squeeze output_dim if it's 1: [batch_size, seq_len, 1] -> [batch_size, seq_len]
            # Labels also need to be [batch_size, seq_len]
            loss = criterion(outputs.squeeze(-1), labels_batch_tensor) # Squeeze the last dim of outputs

            # Backward pass
            loss.backward()

            # Gradient Clipping (optional, helps prevent exploding gradients)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Update weights
            optimizer.step()

            epoch_loss += loss.item()

            if (i + 1) % 1 == 0: # Print every batch (since dataset is small)
                 print(f"Epoch [{epoch+1}/{EPOCHS}], Batch [{i+1}/{num_batches}], Loss: {loss.item():.4f}")

        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
        print(f"--- Epoch [{epoch+1}/{EPOCHS}] completed. Average Training Loss: {avg_epoch_loss:.4f} ---")

        # --- Validation Phase (Placeholder) ---
        model.eval() # Set model to evaluation mode
        val_loss = 0
        num_val_batches = (len(val_sequences) + BATCH_SIZE - 1) // BATCH_SIZE
        
        with torch.no_grad(): # Disable gradient calculations
            for i in range(num_val_batches):
                start_idx = i * BATCH_SIZE
                end_idx = start_idx + BATCH_SIZE
                batch_val_sequences = val_sequences[start_idx:end_idx]
                batch_val_labels_list = val_labels[start_idx:end_idx]

                if not batch_val_sequences:
                    continue

                padded_val_input_batch = pad_sequences(batch_val_sequences, batch_first=True, padding_value=word_to_idx[PAD_TOKEN])
                
                max_len_val_batch = padded_val_input_batch.shape[1]
                padded_val_labels_batch_list = []
                for lbl_seq in batch_val_labels_list:
                    padded_lbl_seq = lbl_seq + [0] * (max_len_val_batch - len(lbl_seq))
                    padded_val_labels_batch_list.append(padded_lbl_seq)

                val_labels_batch_tensor = torch.tensor(padded_val_labels_batch_list, dtype=torch.float).to(device)
                val_input_batch_tensor = padded_val_input_batch.to(device)

                val_outputs = model(val_input_batch_tensor)
                v_loss = criterion(val_outputs.squeeze(-1), val_labels_batch_tensor)
                val_loss += v_loss.item()
        
        avg_val_loss = val_loss / num_val_batches if num_val_batches > 0 else 0
        print(f"--- Epoch [{epoch+1}/{EPOCHS}] Validation Loss: {avg_val_loss:.4f} ---")
        # Add other metrics like accuracy, F1-score here for more detailed validation

    # 7. Save Model
    print("\nTraining complete. Saving model...")
    try:
        # Ensure the directory exists
        model_dir = os.path.dirname(MODEL_SAVE_PATH)
        if not os.path.exists(model_dir) and model_dir: # model_dir can be empty if path is just filename
            os.makedirs(model_dir)
            print(f"Created directory: {model_dir}")
            
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"Model saved to {MODEL_SAVE_PATH}")
    except Exception as e:
        print(f"Error saving model: {e}")


def create_dummy_data_files():
    """Creates dummy data files for testing the training script if they don't exist."""
    print("Checking for dummy data files...")
    data_dir = os.path.join("ComplexTermHighlighter", "data")
    models_dir = os.path.join("ComplexTermHighlighter", "models")

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created directory: {data_dir}")
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print(f"Created directory: {models_dir}")

    if not os.path.exists(TEXT_FILE_PATH):
        print(f"Creating dummy text file (intended for Russian content): {TEXT_FILE_PATH}")
        with open(TEXT_FILE_PATH, 'w', encoding='utf-8') as f:
            f.write("Это простое предложение для тестирования ComplexTermHighlighter.\n") # Example Russian
            f.write("Машинное обучение и искусственный интеллект - увлекательные предметы.\n")
            f.write("Еще одно предложение, чтобы сделать текст немного длиннее и предоставить больше токенов для словаря.\n")
            f.write("Глубокое обучение является подмножеством машинного обучения.")
    else:
        print(f"Text file already exists: {TEXT_FILE_PATH}")

    if not os.path.exists(COMPLEX_TERMS_FILE_PATH):
        print(f"Creating dummy complex terms file (intended for Russian content): {COMPLEX_TERMS_FILE_PATH}")
        with open(COMPLEX_TERMS_FILE_PATH, 'w', encoding='utf-8') as f:
            f.write("машинное обучение\n") # Example Russian
            f.write("искусственный интеллект\n")
            f.write("глубокое обучение\n")
            f.write("простое предложение") 
    else:
        print(f"Complex terms file already exists: {COMPLEX_TERMS_FILE_PATH}")


# --- Entry Point ---
if __name__ == '__main__':
    # Create dummy files if they don't exist, for easier first-time run
    create_dummy_data_files() 
    
    # Check if NLTK 'punkt' is available, as data_loader.py might need it.
    # For Russian, data_loader.py also specifies language='russian' in nltk.word_tokenize,
    # which might depend on specific NLTK Russian models being available.
    try:
        import nltk
        nltk.word_tokenize("test", language='russian') # Test with Russian setting
    except LookupError:
        print("NLTK 'punkt' resource (and potentially Russian models) not found. Downloading...")
        try:
            nltk.download('punkt', quiet=False)
            # nltk.download('russian') # May also be needed for full Russian support in NLTK
            print("'punkt' (and potentially other Russian NLTK resources) downloaded successfully.")
        except Exception as e:
            print(f"Failed to download NLTK resources automatically: {e}")
            print("Please ensure 'punkt' (and 'russian' models if needed) are available for NLTK tokenization.")
    except ImportError:
        print("NLTK is not installed. The data_loader might fail. Please install it (pip install nltk).")


    train_model()
    print("\n`train.py` script execution finished.")
    print("Note: This script uses simplified data loading and splitting for demonstration.")
    print("In a real application, you would use a more extensive dataset and robust data handling.")
