import torch


def seq_distance_loss(input_seq: torch.Tensor, embedded_seq: torch.Tensor, shrink_factor: int = 10) -> torch.Tensor:

    input_seq_diffs = torch.abs(input_seq - input_seq.roll(shifts=1, dims=1))[:, 1:-1] / shrink_factor
    input_seq_diffs = input_seq_diffs.sum(dim=(-1, -2))

    emb_seq_diffs = torch.abs(embedded_seq - embedded_seq.roll(shifts=1, dims=1))[:, 1:-1]
    emb_seq_diffs = emb_seq_diffs.sum(dim=-1)
    distance_loss = torch.sum((emb_seq_diffs - input_seq_diffs) ** 2)

    return distance_loss


def half_window_distance_loss(input_seq: torch.Tensor, embedded_seq: torch.Tensor, shrink_factor: int = 10) -> torch.Tensor:
    """
    sumes both halfs of the seq in input space and greates difference
    same for embedded_seq

    """

    # assert input_seq.shape[0,]
    seq_len = input_seq.shape[1]
    if seq_len % 2 == 1:
        seq_len -= 1
        input_seq = input_seq[:, :-1]
        embedded_seq = embedded_seq[:, :-1]

    seq_half = int(seq_len/2)
    left_input = input_seq[:, :seq_half]
    right_input = input_seq[:, seq_half:]
    input_seq_diffs = torch.abs(left_input - right_input) / shrink_factor
    input_seq_diffs = input_seq_diffs.sum(dim=(-1, -2))

    left_embedd = embedded_seq[:, :seq_half]
    right_embedd = embedded_seq[:, seq_half:]
    emb_seq_diffs = torch.abs(left_embedd - right_embedd)
    emb_seq_diffs = emb_seq_diffs.sum(dim=-1)

    distance_loss = torch.sum((emb_seq_diffs - input_seq_diffs) ** 2)

    return distance_loss


def spectral_centroid_loss(centroids : torch.Tensor, embedded_points : torch.Tensor) -> torch.Tensor:

    return torch.mean(torch.pow(embedded_points[:,-1] - (centroids / 1000), 2))
