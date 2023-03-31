from torch import no_grad
from torch.nn import Module


class BaseModel(Module):
    def init_weights(self):
        pass

    def forward_train(self, *args, **kwargs):
        raise NotImplementedError

    @no_grad()
    def forward_test(self, *args, **kwargs):
        raise NotImplementedError
