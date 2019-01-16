import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def float_to_device(np_arr):
    return torch.from_numpy(np_arr).float().to(DEVICE)

def long_to_device(np_arr):
    return torch.from_numpy(np_arr).long().to(DEVICE)
