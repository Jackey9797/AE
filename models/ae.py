from turtle import forward
import torch 
import torch.nn as nn

# Ctrl shift P + search "interpreter" 选择解释器 
class autoencoder(nn.Module):
    def __init__(self, input_size) -> None:
        super().__init__()
        self.input_size = input_size
        self.R = nn.ReLU()

        # self.input_layer = nn.Linear(input_size, input_size) 
        self.en1 = nn.Linear(input_size, 128)  
        self.en2 = nn.Linear(128, 64)  
        self.de1 = nn.Linear(64, 128)  
        self.de2 = nn.Linear(128, input_size)   

    def forward(self, x): 
        # x = self.R(self.input_layer(x)) 
        x = self.R(self.en1(x))
        x = self.R(self.en2(x))
        x = self.R(self.de1(x))
        x = self.de2(x)
        return x 

# net = ae(500)
# x = torch.rand((24, 500)) 
# print(x.shape)
# print(net(x).shape)