from pyth_imports import *
from STRUCTS.dataset import *

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv  = nn.Sequential(
          nn.Conv2d(in_channels=1, out_channels=769, kernel_size=(1,2), stride=(1,2)),
          nn.ReLU(),
          nn.Conv2d(in_channels=769, out_channels=152, kernel_size=(2,2), stride=(2,2)),
          nn.ReLU(),
          nn.Flatten(),
          nn.Dropout(0.2),
          nn.Linear(29184, 2),
          nn.Softmax(dim=1)
          )

    def forward(self, x):
        x = self.conv(x)
        return x

class MLP(nn.Module):


  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Flatten(),
      nn.Linear(4 * EMBEDDING_SIZE, EMBEDDING_SIZE),
      nn.ReLU(),
      nn.Linear(EMBEDDING_SIZE, EMBEDDING_SIZE/2),
      nn.ReLU(),
      nn.Linear(EMBEDDING_SIZE/2, 2)
    )


  def forward(self, x):

    return self.layers(x)