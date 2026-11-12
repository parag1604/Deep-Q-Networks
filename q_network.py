"""
Step 1 of 4 - the Q-network.

A small MLP maps a state s to one number per action: Q_w(s, a) for all a
at once. One forward pass therefore gives both the value of the action we
took (for the loss) and max_a' Q(s', a') (for the target).
"""
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    def __init__(self, inp_dim, hid_dim, out_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(inp_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, hid_dim)
        self.fc3 = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return self.fc3(x)
