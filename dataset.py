import os
import glob
import cv2
import sys
import torch
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from vocab import vocab_mapping


class Dataset(data.Dataset):
    def __init__(self, num_of_frame, root, mode):
        self.num_of_frame = num_of_frame
        subdir = mode

        if mode == 'train' or 'val':
            video_path = os.path.join(root, subdir, '*', 'lip', '*.mpg')
            align_path = os.path.join(root, subdir, '*', 'word', '*.align')
        elif mode == 'test':
            # seen 으로 할지 unseen 으로 할지는 나중에 선택해서 수정
            video_path = os.path.join(root, subdir, 'seen', '*', 'lip', '*.mpg')
            align_path = os.path.join(root, subdir, 'seen', '*', 'word', '*.align')
        else:
            print('ERROR: CHECK YOUR MODE AGAIN')
            sys.exit()

        self.all_video = sorted(glob.glob(video_path))
        self.all_align = sorted(glob.glob(align_path))
        print('FOUND :', len(self.all_video), 'Video,', len(self.all_align), 'Align')

        for i in range(len(self.all_video)):
            video_base = os.path.basename(self.all_video[i]).replace('.mpg', '')
            align_base = os.path.basename(self.all_align[i]).replace('.align', '')
            if video_base != align_base:
                print('ERROR: VIDEO AND ALIGN NOT MATCH')
                sys.exit()

    def __getitem__(self, index):
        video_path = self.all_video[index]
        align_path = self.all_align[index]
        frames = self.load_video(video_path)
        frame_length = frames.size(1)
        text = self.load_align(align_path)
        words_int = [vocab_mapping[c] for c in text]
        return frames, words_int, text, frame_length

    def __len__(self):
        return len(self.all_video)

    def load_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = torch.FloatTensor(3, self.num_of_frame, 120, 120)
        to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.7136, 0.4906, 0.3283), std=(0.1138, 0.1078, 0.0917))
        ])

        for f in range(self.num_of_frame):
            bgr_img = cap.read()[1]
            if bgr_img is None:
                print('ERROR: SOMETHING WRONG WITH NUM OF FRAME')
                print('FIle -', video_path)
                sys.exit()
            rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(rgb_img)
            frame = to_tensor(frame)
            frames[:, f, :, :] = frame
        return frames

    def load_align(self, align_path):
        with open(align_path, 'r') as f:
            lines = f.readlines()
            words = []
            for line in lines:
                word = line.split(' ')[-1].replace('\n', '')
                if word == 'sil':
                    continue
                char_list = list(word)
                words.extend(char_list)
                words.append(' ')
            words = words[:-1]
        return words


def warpctc_collate(batch):
    frames, words_int, text, frame_length = zip(*batch)
    frames = default_collate(frames)
    y = []
    for t in words_int:
        y.extend(t)
    labels = torch.IntTensor(y)
    label_lens = torch.IntTensor([len(label) for label in words_int])
    frame_lens = torch.IntTensor(frame_length)
    return frames, labels, frame_lens, label_lens, text


def get_loader(dataset, batch_size, shuffle, drop_last):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, collate_fn=warpctc_collate)


if __name__ == '__main__':
    dataset = Dataset(75, '/Users/changdae/Desktop/result',  'train')
    loader = DataLoader(dataset, batch_size=4, shuffle=True, drop_last=True, collate_fn=warpctc_collate)

    for step, (frames, labels, frame_lens, label_lens, text) in enumerate(loader):
        print('==============')
        print(frames.size())  # (N, 3, 75, 120, 120)
        print(labels.shape)  # 1-d tensor containing all the targets of the batch in one large sequence
        print(frame_lens)  # tensor([75, 75, .., 75])
        print(label_lens)  # Tensor of (batch) containing label length of each example
        print(text)  # Tuple containing list of sentences
        break

    '''
    * loader output example
    frames.size -> torch.Size([4, 3, 75, 120, 120])
    labels.size -> torch.Size([101])
    frame_lens -> tensor([75, 75, 75, 75], dtype=torch.int32)
    label_lens -> tensor([25, 24, 29, 23], dtype=torch.int32)
    text -> (['b', 'i', 'n', ' ', ... 'h'] , ['p', 'l', ... , 'e'] , ..)
    '''