from torch.nn import Module

class BaseModel(Module):
    def forward_train(self, *args, **kwargs):
        raise NotImplementedError
    
    def forward_test(self, *args, **kwargs):
        raise NotImplementedError
