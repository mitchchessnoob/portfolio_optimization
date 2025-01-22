
import torch.nn as nn

class LSTM_model():
    """
    Create and configure an LSTM model for stock price prediction.
    """

    def __init__(self, windows_length = 60, num_features = 5, n_units = [50, 50, 50], n_layers = 3, dropout = 0.2):
        """
        Initialize layers
        Input:
        - num_features: number of features in the input data
        - n_units: list containing the number of units for each LSTM layer 
        - n_layers: number of LSTM layers
        - dropout: dropout rate

        """
        assert len(n_units) == n_layers, "The number of units must match the number of layers"
        self.dropout = dropout
        self.layers = [nn.LSTM(units=n_units[i], return_sequences=True, input_shape=(windows_length, num_features)) for i in range(n_layers)]
        self.dense = nn.Dense(units=1)
        self.act = nn.Dense()

    def forward(self, x):
        """
        Forward pass
        Input:
        - x: input data of dimention (batch_size, windows_length, num_features)

        """
        for layer in self.layers:
            x = layer(x)
            x = nn.Dropout(self.dropout)(x)
        x = self.dense(x)

        return x
