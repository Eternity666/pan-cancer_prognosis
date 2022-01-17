import torch
import torch.nn as nn


class ResNet(nn.Module):

    def __init__(self, size, num_layers):

        super(ResNet, self).__init__()

        self.num_layers = num_layers
        self.bn = nn.BatchNorm1d(size)
        self.hidden = nn.Linear(size, size)

    def forward(self, x):

        for i in range(self.num_layers):

            identity = x

            x = self.bn(x)
            x = torch.relu(x)
            x = self.hidden(x)
            x = self.bn(x)
            x = torch.relu(x)
            x = self.hidden(x)

            x = x + identity

        return x


class COXLoss(nn.Module):

    def __init__(self):
        super(COXLoss, self).__init__()

    def forward(self, pred, vital_status):

        hazard_ratio = torch.exp(pred)
        log_risk = torch.log(torch.cumsum(hazard_ratio, dim=0))
        uncensored_likelihood = pred.T - log_risk
        censored_likelihood = uncensored_likelihood * vital_status
        num_observed_events = torch.sum(vital_status)
        neg_likelihood = -torch.sum(censored_likelihood) / num_observed_events

        return neg_likelihood
