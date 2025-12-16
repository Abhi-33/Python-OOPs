import torch
from torch_basic import *

model = NeuralNetwork().to(device)
model.load_state_dict(torch.load("model.pth",weights_only=True))
model.eval()
#model.eval() -> disables training behaviour , correct inference mode