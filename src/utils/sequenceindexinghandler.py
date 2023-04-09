

import numpy as np


def cut_offset_start_end(seq: np.ndarray, offset_percent_start: float, offset_percent_end: float) -> np.ndarray:
    l_seq = len(seq)
    # remove the first and last 10% of the signal to avoid artifacts
    seq = seq[int(round(l_seq*offset_percent_start)):-int(round(l_seq*offset_percent_end))]

    return seq

def extract_sequence(seq: np.ndarray, start_percent: float, end_percent: float) -> np.ndarray:
    l_seq = len(seq)
    # remove the first and last 10% of the signal to avoid artifacts
    seq = seq[int(round(l_seq*start_percent)):int(round(l_seq*end_percent))]

    return seq