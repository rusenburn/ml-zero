import torch as T
from torch.cuda import device
class dotdict(dict):
    def __getattr__(self, name):
        return self[name]


current_device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

def get_device()->device:
    return current_device



    
