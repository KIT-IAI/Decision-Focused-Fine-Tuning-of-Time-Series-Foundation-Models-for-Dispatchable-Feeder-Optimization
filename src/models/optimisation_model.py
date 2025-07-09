from torch import nn
import torch.nn.functional as F
import torch

class OptiEstimator(nn.Module):
    """
    The simple surrogate model for approximating the cost of the optimization problem.
    """

    def __init__(self , output_size):
        super(OptiEstimator, self).__init__()
        self.fc0_1 = nn.Linear(42, 64)
        self.fc0_2 = nn.Linear(42, 64)


        self.fc1 = nn.Linear(133, 64) 
        self.fc2 = nn.Linear(64, 64) 
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 8)
        self.fc6 = nn.Linear(8, output_size)

    def forward(self, input):
        # split the first 42 from the input
        x_0 = input[:,:42]
        # split the second 42 from the input
        x_1 = input[:,42:84]
        #split the last value from the input
        x_2 = input[:,84:]
        # give the first 42 to the first layer
        x_0 = F.selu(self.fc0_1(x_0))
        # give the second 42 to the second layer
        x_1 = F.selu(self.fc0_2(x_1))
        # give the last value to the third layer


        concat = torch.cat((x_0,x_1,x_2),1)

        x = F.selu(self.fc1(concat))
        x = F.selu(self.fc2(x))
        x = F.selu(self.fc3(x))
        x = F.selu(self.fc4(x))
        x = F.selu(self.fc5(x))
        x = torch.sigmoid(self.fc6(x))

        return x
    