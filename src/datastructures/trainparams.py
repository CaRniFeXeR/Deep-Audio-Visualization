from dataclasses import dataclass

@dataclass
class Trainparams:
    n_epochs : int = 7
    learning_rate : float = 0.001
    batch_size : int = 1
    prediction_seq_length : int = 45
    n_elements_pred : int = 1
    seq_prediction_start_epoch : int = 0
    dist_loss_start_epoch : int = 0
    rec_loss : str = "bce"
    seq_loss : str = "mse"
    seq_loss_weight : float = 0.01
    kl_loss_weight : float = 0.001
    dist_loss_weight : float = 0.0001
    use_sprectral_loss : bool = False
