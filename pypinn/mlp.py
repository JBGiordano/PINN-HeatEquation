import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MLP(torch.nn.Module):
    def __init__(self,sizes,k):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.ka = torch.nn.Parameter(data=torch.Tensor([k]), requires_grad=True)
        for i in range(len(sizes)-1):
            self.layers.append(torch.nn.Linear(sizes[i],sizes[i+1]))
    def forward(self,x):
        h = x
        for hidden in self.layers[:-1]:
            h = torch.tanh(hidden(h))
        output = self.layers[-1]
        y = output(h)
        return y
    