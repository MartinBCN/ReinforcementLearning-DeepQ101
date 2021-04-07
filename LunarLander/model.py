import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size: int, action_size: int, seed: int):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, action_size)

        self.drop1 = nn.Dropout(p=0.5)

    def forward(self, state: Tensor) -> Tensor:
        """
        maps state -> action values

        Parameters
        ----------
        state: Tensor

        Returns
        -------
        action: Tensor
        """
        x = self.drop1(F.relu(self.fc1(state)))
        action = self.fc2(x)

        return action
