import os
# Placeholder for NLTK, actual import within tokenize_text
# import nltk
# Placeholder for transformers, actual import within tokenize_text
# from transformers import AutoTokenizer

def load_text_from_file(file_path: str) -> str:
    """
    Reads and returns the text content of the file.

    Args:
        file_path (str): The path to the text file.

    Returns:
        str: The text content of the file.
             Returns an empty string if FileNotFoundError occurs.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return ""

def tokenize_text(text: str, tokenizer_name: str = 'nltk') -> list[str]:
    """
    Tokenizes the input text.

    Args:
        text (str): The text to tokenize.
        tokenizer_name (str): The tokenizer to use ('nltk' or 'transformer').
                              Defaults to 'nltk'.

    Returns:
        list[str]: A list of tokens.
    """
    if tokenizer_name == 'nltk':
        # For NLTK, ensure 'punkt' and specific Russian tokenizer models are downloaded
        # e.g., by running:
        # import nltk
        # nltk.download('punkt')
        # nltk.download('russian') # For Russian-specific sentence tokenization models, if applicable
        import nltk
        try:
            # Added language='russian' for improved Russian tokenization
            tokens = nltk.word_tokenize(text, language='russian')
        except LookupError:
            print("NLTK 'punkt' resource (and potentially 'russian' models) not found.")
            print("Please download them by running: nltk.download('punkt') and potentially nltk.download('russian').")
            # Attempt to download punkt if not found (might fail in restricted environments)
            try:
                nltk.download('punkt', quiet=True)
                # Consider also downloading 'russian' if word_tokenize for Russian needs more than just 'punkt'
                # nltk.download('russian', quiet=True) 
                tokens = nltk.word_tokenize(text, language='russian')
            except Exception as e:
                print(f"Failed to download NLTK resources automatically: {e}")
                return []
        return tokens
    elif tokenizer_name == 'transformer':
        # Placeholder for transformer tokenizer
        # For Russian, consider models like:
        # "DeepPavlov/rubert-base-cased"
        # "sberbank-ai/sbert_large_nlu_ru"
        # from transformers import AutoTokenizer
        # tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased") # Example
        # tokens = tokenizer.tokenize(text)
        print("Transformer tokenizer not implemented yet. Returning empty list.")
        return []
    elif tokenizer_name == 'pymorphy2':
        # Requires pymorphy2 library:
        # import pymorphy2
        # morph = pymorphy2.MorphAnalyzer()
        # tokens = [token.word for token in morph.parse(text)] # Basic tokenization example
        # Note: pymorphy2 provides more detailed morphological analysis which can be leveraged.
        print("Pymorphy2 tokenizer not implemented yet. Returning empty list.")
        return []
    else:
        print(f"Unknown tokenizer: {tokenizer_name}. Returning empty list.")
        return []

def label_tokens(tokens: list[str], complex_terms_list: list[str]) -> list[int]:
    """
    Labels tokens based on a list of complex terms using exact matching.

    # For Russian, simple exact matching is often insufficient due to morphology (word inflections).
    # Consider using lemmatization on both tokens and complex_terms_list before matching,
    # or employ embedding-based similarity for more robust matching.

    Args:
        tokens (list[str]): A list of tokens.
        complex_terms_list (list[str]): A list of known complex terms.

    Returns:
        list[int]: A list of labels (0 or 1) corresponding to each input token.
                   1 if the token is part of a complex term, 0 otherwise.
    """
    labels = [0] * len(tokens)
    num_tokens = len(tokens)
    normalized_complex_terms = [term.lower() for term in complex_terms_list]

    for i in range(num_tokens):
        # Check for multi-word complex terms
        for term in normalized_complex_terms:
            term_tokens = term.split()
            term_len = len(term_tokens)
            if i + term_len <= num_tokens:
                # Extract a slice of tokens from the main list
                token_slice = [token.lower() for token in tokens[i:i + term_len]]
                # Join the slice into a string to compare with the complex term
                # This handles cases where complex terms might have internal punctuation in some tokenizers
                # but here we assume complex_terms_list are pre-split if needed.
                # For simple exact match, we compare lists of tokens.
                if token_slice == term_tokens:
                    for j in range(term_len):
                        labels[i + j] = 1
                    # Skip ahead by term_len -1 because these tokens are now labeled.
                    # The outer loop will increment by 1.
                    # This simple approach might need refinement for overlapping terms.
                    # For now, the first match wins.
                    # Consider i += term_len -1 if we want to avoid re-checking sub-parts of matched terms.
                    # However, this could miss overlapping terms.
                    # Current logic: if "A B C" is complex, and tokens are A B C D,
                    # A, B, C get 1. Then check starts from B, then C, then D.
                    # This is acceptable for a basic version.

    return labels

def load_complex_terms(file_path: str) -> list[str]:
    """
    Reads each line from the file, strips whitespace, and returns a list of complex term strings.

    Args:
        file_path (str): The path to the file containing complex terms (one per line).

    Returns:
        list[str]: A list of complex term strings.
                   Returns an empty list if FileNotFoundError occurs.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Error: Complex terms file not found at {file_path}")
        return []

def preprocess_data(text_file_path: str, complex_terms_file_path: str, tokenizer_name: str = 'nltk'):
    """
    Orchestrates the data preprocessing steps: loading text, loading complex terms,
    tokenizing text, and labeling tokens.

    Args:
        text_file_path (str): Path to the text file.
        complex_terms_file_path (str): Path to the file containing complex terms.
        tokenizer_name (str): The tokenizer to use ('nltk' or 'transformer').

    Returns:
        tuple: A tuple containing:
               - tokens (list[str]): The list of tokens.
               - labels (list[int]): The list of labels for each token.
               Returns ([], []) if any step fails.
    """
    text = load_text_from_file(text_file_path)
    if not text:
        return [], []

    complex_terms = load_complex_terms(complex_terms_file_path)
    if not complex_terms:
        # Depending on requirements, we might proceed with empty complex_terms
        # For now, let's say it's an issue if the file is specified but not found or empty.
        print("Warning: Complex terms list is empty or file not found.")
        # return [], [] # Or proceed with empty list: complex_terms = []

    tokens = tokenize_text(text, tokenizer_name=tokenizer_name)
    if not tokens:
        return [], []

    labels = label_tokens(tokens, complex_terms)
    
    return tokens, labels

if __name__ == '__main__':
    # Example Usage (for testing purposes, will not run in this environment directly)
    # Create dummy files for testing
    print("Creating dummy files for example usage...")
    dummy_text_content = "This is a sample sentence about machine learning and artificial intelligence."
    dummy_complex_terms_content = "machine learning\nartificial intelligence\ndeep learning"
    
    test_dir = "test_data"
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
        
    dummy_text_file = os.path.join(test_dir, "sample_text.txt")
    dummy_complex_file = os.path.join(test_dir, "complex_terms.txt")
    
    with open(dummy_text_file, 'w') as f:
        f.write(dummy_text_content)
    
    with open(dummy_complex_file, 'w') as f:
        f.write(dummy_complex_terms_content)

    print(f"Dummy text file created at: {dummy_text_file}")
    print(f"Dummy complex terms file created at: {dummy_complex_file}")

    print("\n--- Example 1: NLTK Tokenizer ---")
    tokens_nltk, labels_nltk = preprocess_data(dummy_text_file, dummy_complex_file, tokenizer_name='nltk')
    if tokens_nltk:
        print("Tokens:", tokens_nltk)
        print("Labels:", labels_nltk)
        for token, label in zip(tokens_nltk, labels_nltk):
            print(f"{token}: {label}")

    # print("\n--- Example 2: Transformer Tokenizer (Placeholder) ---")
    # tokens_transformer, labels_transformer = preprocess_data(dummy_text_file, dummy_complex_file, tokenizer_name='transformer')
    # if tokens_transformer:
    #     print("Tokens:", tokens_transformer)
    #     print("Labels:", labels_transformer)

    print("\n--- Example 3: File Not Found ---")
    tokens_fnf, labels_fnf = preprocess_data("non_existent_text.txt", "non_existent_complex.txt")
    # Expected output: Error messages and empty lists

    # Clean up dummy files
    # print("\nCleaning up dummy files...")
    # os.remove(dummy_text_file)
    # os.remove(dummy_complex_file)
    # os.rmdir(test_dir)
    # print("Dummy files removed.")
    print("\nNote: NLTK tokenization might require 'punkt' resource.")
    print("If you see an error related to 'punkt', run nltk.download('punkt') in a Python console.")
    print("The example usage within if __name__ == '__main__': is for demonstration and local testing.")
    print("It will attempt to create dummy files and run preprocessing.")
    print("In a restricted environment, file operations or NLTK downloads might be limited.")
