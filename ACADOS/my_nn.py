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
        )

    def forward(self, x):
        out = self.linear_relu_stack(x)
        return out


def NNTrain(my_dataloader, optimizer, model, criterion, n_epoch, val, beta, loss_stop):
    for it in range(n_epoch):  # loop over the dataset multiple times

        for i, data in enumerate(my_dataloader):
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            val = beta * val + (1 - beta) * loss.item()

            if val <= loss_stop:
                return
