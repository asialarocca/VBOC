import torch.nn as nn
import torch
import numpy as np

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
        
class NeuralNetGuess(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetGuess, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        out = self.linear_relu_stack(x)
        return out

class RecurrentNeuralNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(RecurrentNeuralNet, self).__init__()
        
        self.rnn = nn.RNN(input_size, input_size, batch_first=True)
        self.out_size = output_size
        self.in_size = input_size

    def forward(self, x):
        yn = x
        hn = torch.zeros(1,x.size(0),x.size(2))

        outputs = []
        for i in range(self.out_size):
            yn, hn = self.rnn(yn, hn)        # Use previous output and hidden state
            outputs.append(yn)

        output = torch.reshape(torch.cat(outputs),(x.size(0),self.out_size,self.in_size))
        # print(output.size())
        return output
