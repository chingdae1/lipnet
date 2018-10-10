import torch
import torch.nn as nn
import torch.nn.init as init
import math


class LipNet(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.body = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=(3, 8, 8), stride=(1, 2, 2), padding=(1, 2, 2)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.Dropout3d(p=0.5),
            nn.Conv3d(32, 64, kernel_size=(3, 8, 8), stride=(1, 1, 1), padding=(1, 2, 2)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.Dropout3d(p=0.5),
            nn.Conv3d(64, 96, kernel_size=(3, 5, 5), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.Dropout3d(p=0.5)
        )
        self.gru1 = nn.GRU(input_size=(96 * 5 * 5), hidden_size=256, bidirectional=True, batch_first=True)
        self.gru2 = nn.GRU(input_size=512, hidden_size=256, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(in_features=(75 * 512), out_features=(75 * (vocab_size + 1)))

        # Init weight
        for m in self.body.modules():
            if isinstance(m, nn.Conv3d):
                init.kaiming_normal_(m.weight, nonlinearity='relu')
                init.constant_(m.bias, 0)

        for m in (self.gru1, self.gru2):
            stdv = math.sqrt(2 / (96 * 5 * 5 + 256))
            for i in range(0, 256 * 3, 256):
                init.uniform_(m.weight_ih_l0[i: i + 256],
                              -math.sqrt(3) * stdv, math.sqrt(3) * stdv)
                init.orthogonal_(m.weight_hh_l0[i: i + 256])
                init.constant_(m.bias_ih_l0[i: i + 256], 0)
                init.uniform_(m.weight_ih_l0_reverse[i: i + 256],
                              -math.sqrt(3) * stdv, math.sqrt(3) * stdv)
                init.orthogonal_(m.weight_hh_l0_reverse[i: i + 256])
                init.constant_(m.bias_ih_l0_reverse[i: i + 256], 0)

        init.kaiming_normal_(self.fc.weight, nonlinearity='sigmoid')
        init.constant_(self.fc.bias, 0)

    def forward(self, x):
        out = self.body(x)
        out = out.view((-1, 75, 96 * 5 * 5))
        out, _ = self.gru1(out)
        out, _ = self.gru2(out)
        out = out.view((-1, 75 * 512))
        out = self.fc(out)
        out = out.view(-1, 75, self.vocab_size + 1)
        return out  # (N, 75, vocab_size + 1)


if __name__ == '__main__':
    video_stream = torch.zeros((1, 3, 75, 120, 120))
    lipnet = LipNet(27)
    x = lipnet(video_stream)
    print(x.shape)
