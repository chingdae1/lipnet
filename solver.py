import torch
import os
import glob
from dataset import Dataset
from dataset import get_loader
from model import LipNet
from math import sqrt
from warpctc_pytorch import CTCLoss
from CTCdecoder import Decoder

class Solver():
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.train_data = Dataset(num_of_frame=config.num_of_frame,
                                  root=config.data_path,
                                  mode='train')
        self.train_loader = get_loader(dataset=self.train_data,
                                       batch_size=config.batch_size,
                                       shuffle=True,
                                       drop_last=True)
        self.val_data = Dataset(num_of_frame=config.num_of_frame,
                                root=config.data_path,
                                mode='val')
        self.val_loader = get_loader(dataset=self.val_data,
                                     batch_size=config.batch_size,
                                     shuffle=True,
                                     drop_last=True)
        # self.test_data
        self.lipnet = LipNet(config.vocab_size).to(self.device)
        self.ctc_loss = CTCLoss()
        self.optim = torch.optim.Adam(self.lipnet.parameters(),
                                      config.learning_rate)

    def fit(self):
        for epoch in range(1, self.config.epoch + 1):
            for step, (frames, labels, frame_lens, label_lens, text) in enumerate(self.train_loader):
                frames = frames.to(self.device)
                labels = labels
                frame_lens = frame_lens
                label_lens = label_lens
                output = self.lipnet(frames)
                acts = output.permute(1, 0, 2).contiguous().cpu()  # (75, N, 28)
                loss = self.ctc_loss(acts, labels, frame_lens, label_lens)
                loss = loss.mean()

                if not torch.isnan(loss) and -1000000 <= loss <= 1000000:
                    loss.backward()
                    self.optim.step()
                else:
                    print('Skip NaN loss.')
                    continue

                print('Epoch[{}/{}]  Step[{}/{}]  Loss: {:.8f}  LR: {:.8f}'.format(
                    epoch, self.config.epoch, step + 1, self.train_data.__len__() * 2 // self.config.batch_size,
                    loss.item(), self.optim.param_groups[0]['lr']
                ))

            if epoch % self.config.save_every == 0:
                self.save(epoch)

            if epoch > self.config.decay_after and (epoch - self.config.decay_after) % self.config.decay_every == 1:
                self.update_optim()

            self.validation()
            self.lipnet.train()

    def save(self, epoch):
        checkpoint = {
            'net': self.lipnet.state_dict(),
            'config': self.config
        }
        os.makedirs(self.config.save_dir, exist_ok=True)
        output_path = os.path.join(self.config.save_dir, 'model_{}'.format(epoch))
        torch.save(checkpoint, output_path)

    def update_optim(self):
        old_lr = self.optim.param_groups[0]['lr']
        new_lr = old_lr * self.config.decay_rate
        self.optim = torch.optim.Adam(self.lipnet.parameters(),
                                      new_lr)

    def validation(self):
        self.lipnet.eval()
        pass
