import argparse
import os
import sys

# Attempt to import from other modules within the ComplexTermHighlighter package
# This structure assumes ComplexTermHighlighter might be installed or run as a package.
try:
    from .train import train_model as train_model_func
    from .train import TEXT_FILE_PATH as DEFAULT_TRAIN_TEXT_PATH
    from .train import COMPLEX_TERMS_FILE_PATH as DEFAULT_TRAIN_COMPLEX_PATH
    from .train import MODEL_SAVE_PATH as DEFAULT_MODEL_SAVE_PATH_TRAIN
    from .train import EPOCHS as DEFAULT_EPOCHS
    from .train import BATCH_SIZE as DEFAULT_BATCH_SIZE
    from .train import LEARNING_RATE as DEFAULT_LR
    
    from .highlighter import get_highlighted_text
    from .highlighter import MODEL_PATH as DEFAULT_HIGHLIGHT_MODEL_PATH
    from .highlighter import WORD_TO_IDX_PATH as DEFAULT_HIGHLIGHT_VOCAB_PATH
    
    from .data_loader import load_text_from_file
except ImportError:
    print("Error: Could not import modules from ComplexTermHighlighter package.")
    print("Ensure the script is run from the parent directory of ComplexTermHighlighter or the package is installed.")
    # Fallback imports for direct script execution (e.g., python ComplexTermHighlighter/main.py)
    # This requires Python to be able to find these modules, e.g. by being in the same directory
    # or by adjusting PYTHONPATH. For this task, we assume one of these will work.
    try:
        from train import train_model as train_model_func
        from train import TEXT_FILE_PATH as DEFAULT_TRAIN_TEXT_PATH
        from train import COMPLEX_TERMS_FILE_PATH as DEFAULT_TRAIN_COMPLEX_PATH
        from train import MODEL_SAVE_PATH as DEFAULT_MODEL_SAVE_PATH_TRAIN
        from train import EPOCHS as DEFAULT_EPOCHS
        from train import BATCH_SIZE as DEFAULT_BATCH_SIZE
        from train import LEARNING_RATE as DEFAULT_LR

        from highlighter import get_highlighted_text
        from highlighter import MODEL_PATH as DEFAULT_HIGHLIGHT_MODEL_PATH
        from highlighter import WORD_TO_IDX_PATH as DEFAULT_HIGHLIGHT_VOCAB_PATH
        
        from data_loader import load_text_from_file
        print("Successfully used fallback imports for train, highlighter, and data_loader.")
    except ImportError as e:
        print(f"Fallback import failed: {e}")
        print("Please ensure that train.py, highlighter.py, and data_loader.py are in the Python path.")
        sys.exit(1)


def handle_train_args(args):
    """Handles the logic for the 'train' sub-command."""
    print("Starting training process via main.py CLI...")

    # Note: The train_model_func in train.py uses global variables for configuration.
    # A cleaner approach would be to refactor train_model_func to accept all these
    # parameters directly. For now, we might need to override globals in train.py
    # or ensure it's structured to pick up changes made here if possible.
    # This example assumes train_model_func can be called and will use its internal
    # defaults or that we can update them if train.py is structured to allow it.
    # For this task, we'll call it directly, assuming it uses its own defaults primarily,
    # and the CLI args here are more for showing how they *would* be passed.
    
    # To actually use these args, train.py's train_model() would need to be refactored
    # to accept them as parameters. Example of how it *could* work if refactored:
    # train_model_func(
    #     text_file_path=args.text_file,
    #     complex_terms_file_path=args.complex_terms_file,
    #     model_save_dir=args.model_save_dir, # This would require train.py to use this for path construction
    #     epochs=args.epochs,
    #     batch_size=args.batch_size,
    #     learning_rate=args.lr
    # )
    
    # Current train_model_func does not accept these as parameters.
    # We will print the received arguments and call train_model_func.
    # The user would need to modify train.py to use these values.
    print(f"  Text file: {args.text_file}")
    print(f"  Complex terms file: {args.complex_terms_file}")
    print(f"  Model save directory (info only, train.py uses internal path): {args.model_save_dir}")
    print(f"  Epochs (info only, train.py uses internal default): {args.epochs}")
    print(f"  Batch size (info only, train.py uses internal default): {args.batch_size}")
    print(f"  Learning rate (info only, train.py uses internal default): {args.lr}")
    
    print("\nCalling train_model_func from train.py...")
    print("Note: train_model_func will use its own internal path and hyperparameter settings.")
    print("To use CLI arguments for paths/hyperparameters, train.py needs refactoring.")
    
    # A more direct way if train.py's globals were meant to be overridden (less ideal):
    # import train
    # train.TEXT_FILE_PATH = args.text_file
    # train.COMPLEX_TERMS_FILE_PATH = args.complex_terms_file
    # # ... and so on for other parameters.
    # train.train_model()

    try:
        # This will run train_model with its own defined paths and hyperparameters.
        # If train.py's create_dummy_data_files() is called within train_model_func(),
        # it will use paths like 'ComplexTermHighlighter/data/sample_text.txt'.
        # The args.text_file here is effectively ignored by the current train.py structure.
        # For this CLI to fully control paths, train.py would need to be refactored.
        if not os.path.exists(args.text_file):
            print(f"Warning: Provided text_file '{args.text_file}' does not exist. "
                  "train_model_func might rely on its internal dummy file creation if this path is used.")
        if not os.path.exists(args.complex_terms_file):
            print(f"Warning: Provided complex_terms_file '{args.complex_terms_file}' does not exist. "
                  "train_model_func might rely on its internal dummy file creation if this path is used.")

        train_model_func() # Call the training function
        print("\nTraining process finished.")
        print(f"Check {DEFAULT_MODEL_SAVE_PATH_TRAIN} (or similar path in train.py) for the model.")
    except Exception as e:
        print(f"An error occurred during training: {e}")
        print("Ensure that the training environment is correctly set up and all paths in train.py are valid.")


def handle_highlight_args(args):
    """Handles the logic for the 'highlight' sub-command."""
    print("Starting highlighting process...")

    # Validate required file paths
    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found at '{args.input_file}'")
        return
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at '{args.model_path}'")
        return
    if not os.path.exists(args.vocab_path):
        print(f"Error: Vocabulary file not found at '{args.vocab_path}'")
        return

    print(f"  Input file: {args.input_file}")
    print(f"  Model path: {args.model_path}")
    print(f"  Vocab path: {args.vocab_path}")
    if args.output_file:
        print(f"  Output file: {args.output_file}")

    # 1. Load text from the input file
    text_content = load_text_from_file(args.input_file)
    if not text_content:
        print(f"Error: Could not read text from '{args.input_file}' or file is empty.")
        return

    # 2. Get highlighted text using the function from highlighter.py
    try:
        highlighted_text = get_highlighted_text(
            text_input=text_content,
            model_path=args.model_path,
            vocab_path=args.vocab_path
        )
    except Exception as e:
        print(f"An error occurred during highlighting: {e}")
        return

    # 3. Output the result
    if args.output_file:
        try:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                f.write(highlighted_text)
            print(f"\nHighlighting complete. Output saved to: {args.output_file}")
        except IOError as e:
            print(f"Error: Could not write to output file '{args.output_file}': {e}")
    else:
        print("\n--- Highlighted Text ---")
        print(highlighted_text)
        print("--- End of Highlighted Text ---")


def main():
    """Main function to parse arguments and dispatch to handlers."""
    parser = argparse.ArgumentParser(description="Complex Term Highlighter CLI")
    subparsers = parser.add_subparsers(dest="mode", required=True, help="Mode of operation: 'train' or 'highlight'")

    # --- Train Mode Subparser ---
    # Users should refer to train.py for actual paths/hyperparams used if not refactored.
    train_parser = subparsers.add_parser("train", help="Train a new complex term highlighting model.")
    train_parser.add_argument(
        "--text_file", 
        type=str, 
        required=True, # Make it required to show intent, even if train.py needs refactor
        help=f"Path to the input Russian text file or directory for training. "
             f"(Note: current train.py may use its own default path like '{DEFAULT_TRAIN_TEXT_PATH}')"
    )
    train_parser.add_argument(
        "--complex_terms_file", 
        type=str, 
        required=True, # Make it required to show intent
        help=f"Path to the file containing the list of complex Russian terms. "
             f"(Note: current train.py may use its own default path like '{DEFAULT_TRAIN_COMPLEX_PATH}')"
    )
    train_parser.add_argument(
        "--model_save_dir", 
        type=str, 
        default=os.path.dirname(DEFAULT_MODEL_SAVE_PATH_TRAIN), # Default from train.py's structure (e.g., 'ComplexTermHighlighter/models')
        help=f"Directory to save the trained Russian model and vocabulary. "
             f"(Default: '{os.path.dirname(DEFAULT_MODEL_SAVE_PATH_TRAIN)}'). "
             f"Note: train.py currently uses a fixed path structure like 'term_highlighter_model_russian.pt' within this."
    )
    train_parser.add_argument(
        "--epochs", 
        type=int, 
        default=DEFAULT_EPOCHS, 
        help=f"Number of training epochs. (Default: {DEFAULT_EPOCHS}). "
             f"Note: train.py uses its own internal default."
    )
    train_parser.add_argument(
        "--batch_size", 
        type=int, 
        default=DEFAULT_BATCH_SIZE, 
        help=f"Batch size for training. (Default: {DEFAULT_BATCH_SIZE}). "
             f"Note: train.py uses its own internal default."
    )
    train_parser.add_argument(
        "--lr", 
        type=float, 
        default=DEFAULT_LR, 
        help=f"Learning rate. (Default: {DEFAULT_LR}). "
             f"Note: train.py uses its own internal default."
    )
    train_parser.set_defaults(func=handle_train_args)

    # --- Highlight Mode Subparser ---
    highlight_parser = subparsers.add_parser("highlight", help="Highlight complex terms in a text file using a pre-trained model.")
    highlight_parser.add_argument(
        "--input_file", 
        type=str, 
        required=True, 
        help="Path to the Russian .txt file to process."
    )
    highlight_parser.add_argument(
        "--model_path", 
        type=str, 
        default=DEFAULT_HIGHLIGHT_MODEL_PATH, # Expected to be '.../term_highlighter_model_russian.pt'
        help=f"Path to the pre-trained Russian model file. (Default: '{DEFAULT_HIGHLIGHT_MODEL_PATH}')"
    )
    highlight_parser.add_argument(
        "--vocab_path", 
        type=str, 
        default=DEFAULT_HIGHLIGHT_VOCAB_PATH, # Expected to be '.../word_to_idx_russian.json'
        help=f"Path to the Russian vocabulary file (e.g., word_to_idx_russian.json). (Default: '{DEFAULT_HIGHLIGHT_VOCAB_PATH}')"
    )
    highlight_parser.add_argument(
        "--output_file", 
        type=str, 
        default=None, 
        help="Optional path to save the highlighted text. If not provided, prints to console."
    )
    highlight_parser.set_defaults(func=handle_highlight_args)

    args = parser.parse_args()
    args.func(args) # Call the appropriate handler function (handle_train_args or handle_highlight_args)

if __name__ == '__main__':
    # Note: If running this script directly, ensure that the ComplexTermHighlighter directory
    # is in the PYTHONPATH or that you are in its parent directory, so that
    # `from .train import ...` or `from train import ...` can work.
    # Example: python -m ComplexTermHighlighter.main train --text_file ...
    # Or if ComplexTermHighlighter is in PYTHONPATH: python ComplexTermHighlighter/main.py train --text_file ...
    main()
