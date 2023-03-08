import torch.nn as nn

# Fully connected neural network
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, output_size),
            # nn.Sigmoid(),
        )

    def forward(self, x):
        out = self.linear_relu_stack(x)
        return out
        
class NeuralNetRegression(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetRegression, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.BatchNorm1d(output_size),
        )

        self.linear_relu_stack_in = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
        )
        self.linear_relu_stack_mid = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
        )
        self.linear_relu_stack_out = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.BatchNorm1d(output_size),
        )
        
        self.linear_relu_bridge = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.BatchNorm1d(output_size),
        )

        self.linear_relu_bridge_in = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
        )
        self.linear_relu_bridge_mid = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
        )
        self.linear_relu_bridge_out = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            nn.BatchNorm1d(output_size),
        )

        self.out_relu = nn.ReLU()

    def forward(self, x):
        # out = self.linear_relu_stack(x) #+ self.linear_relu_bridge(x)
        # out = self.out_relu(out)

        out = self.linear_relu_stack_in(x) #+ self.linear_relu_bridge_in(x)
        out = self.out_relu(out)
        out = self.linear_relu_stack_mid(out) #+ self.linear_relu_bridge_mid(out)
        out = self.out_relu(out)
        # out = self.linear_relu_stack_mid(out) #+ self.linear_relu_bridge_mid(out)
        # out = self.out_relu(out)
        # out = self.linear_relu_stack_mid(out) #+ self.linear_relu_bridge_mid(out)
        # out = self.out_relu(out)
        out = self.linear_relu_stack_out(out) #+ self.linear_relu_bridge_out(out)
        out = self.out_relu(out)

        return out
