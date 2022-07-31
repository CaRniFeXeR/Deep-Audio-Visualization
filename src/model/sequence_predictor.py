import torch

from src.datastructures.predictionheadconfig import SeqPredictorConfig


class SequencePredictor(torch.nn.Module):

    def __init__(self, config: SeqPredictorConfig) -> None:
        super().__init__()
        self.config = config

        self.lstm = torch.nn.LSTM(input_size=3, hidden_size=16, num_layers = self.config.n_layers, batch_first=True)
        self.lin_out = torch.nn.Linear(16, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, (hn, cn) = self.lstm(x)

        return self.lin_out(hn[-1])
