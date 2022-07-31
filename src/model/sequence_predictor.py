import torch

from src.datastructures.predictionheadconfig import SeqPredictorConfig 

class SequencePredictor(torch.nn.Module):

    def __init__(self, config : SeqPredictorConfig) -> None:
        super().__init__()
        self.config = config

        self.lstm = torch.nn.LSTM(16)
        self.lin_out = torch.nn.Linear(16,3)

    def forward(self, x):
        lstm_out = self.lstm(x)

        return self.lin_out(lstm_out)