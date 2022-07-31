from dataclasses import dataclass

@dataclass
class Trainparams:
    n_epochs : int = 7
    learning_rate : float = 0.001