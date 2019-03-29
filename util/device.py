import torch


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def float_to_device(np_arr):
    return torch.from_numpy(np_arr).float().to(DEVICE)


def long_to_device(np_arr):
    return torch.from_numpy(np_arr).long().to(DEVICE)


def tensor_float(x):
    if is_tensor(x):
        return x
    return torch.tensor(x, device=DEVICE, dtype=torch.float32)


def tensor_long(x):
    if is_tensor(x):
        return x
    return torch.tensor(x, device=DEVICE, dtype=torch.long)


def is_tensor(x):
    return isinstance(x, torch.Tensor)


def to_np(t):
    return t.cpu().data.numpy()