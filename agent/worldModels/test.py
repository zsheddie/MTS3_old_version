import torch

a = torch.zeros(3,4,5)
b = torch.zeros(3,6,5)
m = []
m.append(a)
m.append(b)
c = torch.cat((a,b), dim = 1)
g = torch.cat(m,dim=1)
print(g.shape)

for i in range (7):
    print(i)