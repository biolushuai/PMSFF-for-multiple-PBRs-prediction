import torch
from torch import nn

from configs import DefaultConfig
configs = DefaultConfig()


class AttentionLayer(nn.Module):
    def __init__(self, feature_dim, padding_size):
        super(AttentionLayer, self).__init__()
        input_dim = feature_dim
        self.padding_size = padding_size
        self.padding = nn.ZeroPad2d(padding=(0, 0, self.padding_size, self.padding_size))
        self.w = nn.Linear(input_dim, configs.attention_hidden_dim, bias=False)
        self.u = nn.Linear(input_dim, configs.attention_hidden_dim, bias=False)
        self.v = nn.Linear(configs.attention_hidden_dim, 1, bias=False)

    def forward(self, x):
        w_length = self.padding_size * 2
        x_length = len(x)
        x_center = x.unsqueeze(1).repeat(1, w_length, 1)

        x_padded = self.padding(x)
        x_local = list()
        for i in range(x_length):
            x_local.append(x_padded[i: i + self.padding_size])
            x_local.append(x_padded[i + self.padding_size + 1: i + self.padding_size * 2 + 1])
        x_local = torch.reshape(torch.cat(x_local, 0), [x_length, w_length, -1])

        energy = torch.tanh(torch.add(self.w(x_center), self.u(x_local)))
        attention = self.v(energy).squeeze(2)
        attention_out = torch.softmax(attention, dim=1).unsqueeze(1)

        g = torch.bmm(attention_out, x_local).squeeze(1)
        return g
