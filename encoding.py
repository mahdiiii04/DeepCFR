import torch

HISTORY_MAP = {
    "": 0, "c": 1, "b": 2,
    "cc": 3, "cb": 4, "bc": 5,
    "cbc": 6, "bf": 7, "cbf": 8
}

def encode_infoset(card, history):
    return torch.tensor(
        [card, HISTORY_MAP[history]],
        dtype=torch.float32
    )
