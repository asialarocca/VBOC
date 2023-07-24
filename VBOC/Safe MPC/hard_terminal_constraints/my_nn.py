import torch.nn as nn

class NeuralNetRegression(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetRegression, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.linear_relu_stack(x)
        return out
    