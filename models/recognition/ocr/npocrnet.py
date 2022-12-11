from typing import Any, List

import torch
import torch.nn as nn
from torchvision.models import resnet18


class BlockRNN(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, bidirectional):
        super(BlockRNN, self).__init__()
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.bidirectional = bidirectional
        # layers
        self.gru = nn.LSTM(
            in_size, hidden_size, bidirectional=bidirectional, batch_first=True
        )

    def forward(self, batch, add_output=False):
        """
        in array:
            batch - [seq_len , batch_size, in_size]
        out array:
            out - [seq_len , batch_size, out_size]
        """
        outputs, hidden = self.gru(batch)
        out_size = int(outputs.size(2) / 2)
        if add_output:
            outputs = outputs[:, :, :out_size] + outputs[:, :, out_size:]
        return outputs


class NPOcrNet(nn.Module):
    def __init__(
        self,
        letters: List = None,
        letters_max: int = 0,
        max_plate_length: int = 8,
        learning_rate: float = 0.02,
        hidden_size: int = 32,
        bidirectional: bool = True,
        label_converter: Any = None,
        val_dataset: Any = None,
        weight_decay: float = 1e-5,
        momentum: float = 0.9,
        clip_norm: int = 5,
    ):
        super().__init__()

        self.letters = letters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.clip_norm = clip_norm
        self.momentum = momentum
        self.max_plate_length = max_plate_length

        self.hidden_size = hidden_size
        self.bidirectional = bidirectional

        self.label_converter = label_converter

        # convolutions
        resnet = resnet18(pretrained=True)
        modules = list(resnet.children())[:-3]
        self.resnet = nn.Sequential(*modules)

        # RNN + Linear
        self.linear1 = nn.Linear(1024, 512)
        self.gru1 = BlockRNN(512, hidden_size, hidden_size, bidirectional=bidirectional)
        self.gru2 = BlockRNN(
            hidden_size, hidden_size, letters_max, bidirectional=bidirectional
        )
        self.linear2 = nn.Linear(hidden_size * 2, letters_max)

        self.automatic_optimization = True
        self.criterion = None
        self.val_dataset = val_dataset
        self.train_losses = []
        self.val_losses = []

    def forward(self, batch: torch.float64):
        """
        ------:size sequence:------
        torch.Size([batch_size, 3, 64, 128]) -- IN:
        torch.Size([batch_size, 16, 16, 32]) -- CNN blocks ended
        torch.Size([batch_size, 32, 256]) -- permuted
        torch.Size([batch_size, 32, 32]) -- Linear #1
        torch.Size([batch_size, 32, 512]) -- IN GRU
        torch.Size([batch_size, 512, 512]) -- OUT GRU
        torch.Size([batch_size, 32, vocab_size]) -- Linear #2
        torch.Size([32, batch_size, vocab_size]) -- :OUT
        """
        batch_size = batch.size(0)

        # convolutions
        batch = self.resnet(batch)

        # make sequences of image features
        batch = batch.permute(0, 3, 1, 2)
        n_channels = batch.size(1)
        batch = batch.reshape(batch_size, n_channels, -1)

        batch = self.linear1(batch)

        # rnn layers
        batch = self.gru1(batch, add_output=True)
        batch = self.gru2(batch)
        # output
        batch = self.linear2(batch)
        batch = batch.permute(1, 0, 2)
        return batch
