import torch
import torch.nn as nn

class SimpleLSTMagger(nn.Module):
    """
    A simple LSTM-based sequence tagger.
    This model takes a sequence of token indices, passes them through an embedding layer,
    an LSTM layer, and a linear layer to produce tag scores for each token.
    """
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, output_dim: int, 
                 n_layers: int = 1, dropout: float = 0.2):
        """
        Initializes the layers of the SimpleLSTMagger model.

        Args:
            vocab_size (int): The size of the vocabulary (number of unique tokens). For Russian, this would be the size of the Russian vocabulary.
            embedding_dim (int): The dimension of the token embeddings.
            hidden_dim (int): The dimension of the LSTM hidden states.
            output_dim (int): The dimension of the output for each token (e.g., number of tags).
                              For binary classification (complex/not complex), this could be 2 
                              (if using CrossEntropyLoss with class indices) or 1 
                              (if using BCEWithLogitsLoss).
            n_layers (int, optional): The number of LSTM layers. Defaults to 1.
            dropout (float, optional): The dropout rate to be applied. Defaults to 0.2.
                                       Dropout is applied to embeddings and LSTM outputs.
                                       If n_layers > 1, LSTM also has dropout between layers.
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dropout_rate = dropout

        # 1. Embedding Layer
        # Converts input token indices into dense vector representations (embeddings).
        # Shape: vocab_size (number of unique words) x embedding_dim (size of embedding vector for each word)
        # Consider loading pre-trained Russian word embeddings here (e.g., from fastText or Word2Vec trained on Russian corpora)
        # and potentially freezing them during initial training stages.
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # 2. Dropout Layer for Embeddings
        # Applied to the output of the embedding layer to prevent overfitting.
        self.embedding_dropout = nn.Dropout(dropout)

        # 3. LSTM Layer
        # Processes the sequence of embeddings, capturing contextual information.
        # - input_size: embedding_dim (size of the input features for each token)
        # - hidden_size: hidden_dim (number of features in the hidden state)
        # - num_layers: n_layers (number of recurrent layers)
        # - batch_first=True: expects input tensors in the shape [batch_size, seq_len, features]
        # - dropout: if n_layers > 1, adds dropout between LSTM layers (not on the last layer's output).
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True, # Crucial for [batch_size, seq_len, features] input format
            dropout=dropout if n_layers > 1 else 0 # LSTM internal dropout
        )

        # 4. Dropout Layer for LSTM Output
        # Applied to the output of the LSTM layer before the final linear layer.
        self.lstm_dropout = nn.Dropout(dropout)

        # 5. Linear Layer (Fully Connected Layer)
        # Maps the LSTM output features for each token to the desired output dimension (e.g., tag scores).
        # - in_features: hidden_dim (size of input from LSTM for each token)
        # - out_features: output_dim (number of scores per token, e.g., 2 for binary tags or 1 for BCEWithLogits)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text_indices: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the model.

        Args:
            text_indices (torch.Tensor): A batch of input token indices.
                                         Shape: [batch_size, seq_len]

        Returns:
            torch.Tensor: The output of the model (tag scores for each token in the sequence).
                          Shape: [batch_size, seq_len, output_dim]
        """
        # text_indices shape: [batch_size, seq_len]
        # Example: [[1, 2, 3, 0], [4, 5, 0, 0]] where 0 is padding.

        # 1. Pass through Embedding Layer
        # embedded shape: [batch_size, seq_len, embedding_dim]
        embedded = self.embedding(text_indices)

        # 2. Apply Dropout to Embeddings
        # embedded_dropped shape: [batch_size, seq_len, embedding_dim]
        embedded_dropped = self.embedding_dropout(embedded)

        # 3. Pass through LSTM Layer
        # lstm_out shape: [batch_size, seq_len, hidden_dim] (if batch_first=True)
        # hidden shape: (tuple of two tensors for hidden and cell states)
        #   - h_n shape: [n_layers * num_directions, batch_size, hidden_dim]
        #   - c_n shape: [n_layers * num_directions, batch_size, hidden_dim]
        # We only need lstm_out for token-level predictions.
        lstm_out, (hidden, cell) = self.lstm(embedded_dropped)
        
        # 4. Apply Dropout to LSTM outputs
        # lstm_out_dropped shape: [batch_size, seq_len, hidden_dim]
        lstm_out_dropped = self.lstm_dropout(lstm_out)

        # 5. Pass LSTM outputs through Linear Layer
        # The linear layer is applied to each time step (token) independently.
        # predictions shape: [batch_size, seq_len, output_dim]
        predictions = self.fc(lstm_out_dropped)

        return predictions

if __name__ == '__main__':
    # Example of how to instantiate the model (for testing/demonstration)
    # This part will not be executed by the autograder but is useful for local testing.
    
    # Define some hyperparameters
    VOCAB_SIZE = 1000  # Example vocabulary size
    EMBEDDING_DIM = 100 # Dimension of word embeddings
    HIDDEN_DIM = 128   # Dimension of LSTM hidden states
    OUTPUT_DIM = 2     # Number of output tags (e.g., 0 for not complex, 1 for complex)
    N_LAYERS = 2       # Number of LSTM layers
    DROPOUT = 0.25

    print("Instantiating SimpleLSTMagger model...")
    model = SimpleLSTMagger(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, DROPOUT)
    print("Model instantiated successfully.")
    print(model)

    # Create a dummy input batch (batch_size=3, seq_len=10)
    batch_size = 3
    seq_len = 10
    dummy_input = torch.randint(0, VOCAB_SIZE, (batch_size, seq_len)) # Random token indices
    print(f"\nDummy input shape: {dummy_input.shape}")

    # Perform a forward pass
    print("Performing a forward pass with dummy input...")
    try:
        output = model(dummy_input)
        print(f"Output shape: {output.shape}") # Expected: [batch_size, seq_len, OUTPUT_DIM]
        assert output.shape == (batch_size, seq_len, OUTPUT_DIM)
        print("Forward pass successful and output shape is correct.")
    except Exception as e:
        print(f"Error during forward pass: {e}")

    # Example for output_dim = 1 (e.g., for BCEWithLogitsLoss)
    OUTPUT_DIM_BCE = 1
    print(f"\nInstantiating SimpleLSTMagger model for BCEWithLogitsLoss (output_dim={OUTPUT_DIM_BCE})...")
    model_bce = SimpleLSTMagger(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM_BCE, N_LAYERS, DROPOUT)
    print("Model for BCE instantiated successfully.")
    print(model_bce)
    try:
        output_bce = model_bce(dummy_input)
        print(f"Output shape for BCE model: {output_bce.shape}") # Expected: [batch_size, seq_len, OUTPUT_DIM_BCE]
        assert output_bce.shape == (batch_size, seq_len, OUTPUT_DIM_BCE)
        print("Forward pass successful for BCE model and output shape is correct.")
    except Exception as e:
        print(f"Error during forward pass for BCE model: {e}")
