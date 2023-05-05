import torch

torch.manual_seed(42)
bsz = 4
num_points = 2
dim = 3

blah = torch.randn((bsz, num_points, dim))

print(blah)
encoded = torch.concatenate((torch.cos(torch.pi*blah), torch.sin(torch.pi*blah)), dim=2)
print(encoded)
print(encoded.shape)
kernel = []
for i in range(2):
    kernel.append(torch.cos((2**i) * torch.pi* blah))
    kernel.append(torch.sin((2**i) * torch.pi* blah))
print(len(kernel))
encoded = torch.concatenate(kernel, dim=2)
print(encoded)
print(encoded.shape)