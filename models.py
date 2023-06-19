import torch
import torch.nn as nn

from layers import AttentionLayer
from configs import DefaultConfig
configs = DefaultConfig()


class A3C3GRUModel(nn.Module):
    def __init__(self):
        super(A3C3GRUModel, self).__init__()
        feature_dim = configs.feature_dim
        window_padding_sizes = configs.window_padding_sizes

        cnn_in_dim = configs.feature_dim * 4
        cnn_out_dim = configs.out_channels
        kernel_sizes = configs.kernel_sizes
        padding1 = (kernel_sizes[0] - 1) // 2
        padding2 = (kernel_sizes[1] - 1) // 2
        padding3 = (kernel_sizes[2] - 1) // 2

        rnn_layer = configs.rnn_hidden_layer
        rnn_in_dim = configs.out_channels * 3
        rnn_out_dim = configs.rnn_hidden_dim

        mlp_in_dim = rnn_out_dim * 2
        mlp_dim = configs.mlp_dim
        dropout_rate = configs.dropout_rate

        self.attention1 = AttentionLayer(feature_dim, window_padding_sizes[0])
        self.attention2 = AttentionLayer(feature_dim, window_padding_sizes[1])
        self.attention3 = AttentionLayer(feature_dim, window_padding_sizes[2])

        self.cnn1 = nn.Conv1d(in_channels=cnn_in_dim, out_channels=cnn_out_dim, kernel_size=kernel_sizes[0], padding=padding1)
        self.cnn2 = nn.Conv1d(in_channels=cnn_in_dim, out_channels=cnn_out_dim, kernel_size=kernel_sizes[1], padding=padding2)
        self.cnn3 = nn.Conv1d(in_channels=cnn_in_dim, out_channels=cnn_out_dim, kernel_size=kernel_sizes[2], padding=padding3)

        self.pool = nn.MaxPool1d(kernel_size=5, stride=1, padding=2, dilation=1)

        self.rnn = nn.GRU(input_size=rnn_in_dim, hidden_size=rnn_out_dim, num_layers=rnn_layer, batch_first=True, bidirectional=True)

        self.linear1 = nn.Sequential(
            nn.Linear(mlp_in_dim, mlp_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        self.linear2 = nn.Sequential(
            nn.Linear(mlp_dim, 1),
            nn.Sigmoid(),
            nn.Dropout(dropout_rate)
        )

    def forward(self, vertex):
        vertex1 = self.attention1(vertex)
        vertex2 = self.attention2(vertex)
        vertex3 = self.attention3(vertex)
        vertex = torch.cat((vertex, vertex1, vertex2, vertex3), dim=1)

        out = torch.unsqueeze(vertex, 0)
        out = out.permute(0, 2, 1)

        out1 = self.cnn1(out)
        out2 = self.cnn2(out)
        out3 = self.cnn3(out)
        out = torch.cat((out1, out2, out3), dim=1)
        out = out.permute(0, 2, 1)

        out = self.pool(out)

        out, _ = self.rnn(out)

        out = self.linear1(out)
        out = self.linear2(out)
        out = torch.squeeze(out)
        return out
