print("Hello, World!")

import torch
print(torch.cuda.is_available())  # True if CUDA is available
print(torch.cuda.device_count())  # Number of CUDA devices
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No CUDA device")


def __init__(self):
    pass