import torch.nn as nn


class NeuralNet(nn.Module):

    def __init__(
        self, input_size,
        class_count,
        num_hid_neurons,
        act='relu'
    ):

        super(NeuralNet, self).__init__()

        if len(num_hid_neurons) == 0:
            num_hid_neurons = [input_size / 2]

        layers = []

        # first hidden layer
        layers.append(nn.Linear(input_size, num_hid_neurons[0]))
        layers.append(self.get_activation(act))

        # other hidden layers
        for i in range(len(num_hid_neurons) - 1):
            input_size = num_hid_neurons[i]
            output_size = num_hid_neurons[i + 1]
            layers.append(nn.Linear(input_size, output_size))
            layers.append(self.get_activation(act))

        # last layer
        layers.append(nn.Linear(num_hid_neurons[-1], class_count))
        self.main = nn.Sequential(*layers)

    def get_activation(self, act='relu'):

        activation = nn.ReLU(inplace=True)
        if act == 'sigmoid':
            activation = nn.Sigmoid()
        elif act == 'tanh':
            activation = nn.Tanh()

        return activation

    def forward(self, x):
        return self.main(x)
