import numpy as np
import torch
import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, hidden_sizes=[64, 64], hidden_action_size=32, hidden_state_size=32):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.state_size = state_size
        self.hidden_sizes = hidden_sizes
        self.action_size = action_size

        self.fc_input = nn.Sequential(nn.Linear(state_size, hidden_sizes[0]), nn.ReLU())

        self.fc_hidden = None
        if len(hidden_sizes)>1:
            hidden_layers = []
            for i in range(len(hidden_sizes)-1):
                layer = nn.Sequential(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]), nn.ReLU())
                hidden_layers.append(layer)
            self.fc_hidden = nn.Sequential(*hidden_layers)

        self.fc_state_value = nn.Sequential(nn.Linear(hidden_sizes[-1], hidden_state_size),
                                             nn.ReLU(),
                                             nn.Linear(hidden_state_size, 1)
                                            )
        self.fc_action_values = nn.Sequential(nn.Linear(hidden_sizes[-1], hidden_action_size),
                                               nn.ReLU(),
                                               nn.Linear(hidden_action_size, action_size)
                                              )
    def forward(self, state):
        x = self.fc_input(state)
        x = self.fc_hidden(x) if self.fc_hidden else x
        state_value = self.fc_state_value(x)
        action_values = self.fc_action_values(x)
        return state_value+action_values
