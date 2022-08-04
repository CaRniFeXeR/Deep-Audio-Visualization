from dataclasses import dataclass

@dataclass
class Trainparams:
    n_epochs : int = 7
    learning_rate : float = 0.001
    batch_size : int = 1
    prediction_seq_length : int = 45
    seq_prediction_start_epoch : int = 0
    rec_loss : str = "bce"
    seq_pred_loss : str = "mse"