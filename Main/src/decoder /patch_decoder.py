import torch 
import torch.nn as nn 
class PatchDecoder(nn.Module):
    def __init__(self, input_dim:int, patch_size:int = 64):
        super().__init__()
        self.patch_size = patch_size
        self.network = nn.Sequential(
            nn.Linear(input_dim , 128 ) , 
            nn.ReLU(),
            nn.Linear(128 , 256),
            nn.ReLU(), 
            nn.Linear(256 , patch_size*patch_size) , 
            nn.Sigmoid()
        )
    def forward(self , observables: torch.Tensor , positional_encoding : torch.Tensor):
            x = torch.cat([observables , positional_encoding ] , dims = 1)
            x = self.network(x)
            return x.view(-1 , 1 , self.patch_size , self.patch_size)
