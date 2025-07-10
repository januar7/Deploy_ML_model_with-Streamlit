import torch

a = torch.load('model_NN_simple.pth')
print('model param',a)
print(a.shape)