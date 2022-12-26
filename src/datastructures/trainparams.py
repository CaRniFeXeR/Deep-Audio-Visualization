from dataclasses import dataclass

@dataclass
class Trainparams:
    n_epochs : int = 7
    learning_rate : float = 0.001
    batch_size : int = 1
    prediction_seq_length : int = 45
    seq_prediction_start_epoch : int = 0
    dist_loss_start_epoch : int = 0
    rec_loss : str = "bce"
    seq_loss : str = "mse"
    seq_loss_weight : float = 0.01
    dist_loss_weight : float = 0.001
