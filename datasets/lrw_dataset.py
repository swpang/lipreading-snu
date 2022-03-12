import torch
import numpy as np
import os
import glob
import librosa
import random
from collections import defaultdict
from datasets.base_dataset import BaseDataset
from utils.dataloaders import get_preprocessing_pipelines
from utils.util import read_txt_lines

class LRWDataset(BaseDataset):
    # dataloader for LRW dataset
    name = 'lrw'

    def __init__(self, config: dict, mode: str):
        BaseDataset.__init__(self, config, mode)
        self.summary_name = self.name
        self.data_dir = config['data_dir']
        self.label_path = config['label_path']
        self.annotation_dir = config.get('annotation_dir')
        # data augmentation
        self.preprocessing = get_preprocessing_pipelines(config['modality'])[mode]

        self.fps = 25 if config['modality'] == "video" else 16000
        self.label_idx = -3
        self._data_files = []
        self._labels = read_txt_lines(self.label_path)

        # -- add examples to self._data_files
        # get npy/npz/mp4 files
        for suffix in ['npz', 'npy', 'mp4']:
            self._data_files.extend(
                glob.glob(os.path.join(self.data_dir, '*', mode, '*.{}'.format(suffix)))
            )
        # If we are not using the full set of labels, remove examples for labels not used
        self._data_files = [f for f in self._data_files if f.split('/')[self.label_idx] in self._labels]
        # -- from self._data_files to self.list
        self.list = defaultdict(list)
        self.instance_ids = defaultdict(list)
        for i, x in enumerate(self._data_files):
            label = x.split('/')[self.label_idx]
            self.list[i] = [x, self._labels.index(label)]
            self.instance_ids[i] = os.path.splitext(x.split('/')[-1])[0]

    def __getitem__(self, idx):
        if self.config['overfit_one_ex'] is not None:
            idx = self.config['overfit_one_ex']
        
        filename = self.list[idx][0]
        if filename.endswith('npz'):
            raw_data = np.load(filename)['data']
        elif filename.endswith('mp4'):
            raw_data = librosa.load(filename, sr=16000)[0][-19456:]
        else:
            raw_data = np.load(filename)

        # -- perform variable length on training set
        if (self.mode == 'train'):
            # read info txt file (to see duration of word, to be used to do temporal cropping)
            info_txt = os.path.join(self.annotation_dir, *filename.split('/')[self.label_idx:])  # swap base folder
            info_txt = os.path.splitext(info_txt)[0] + '.txt'  # swap extension
            info = read_txt_lines(info_txt)

            utterance_duration = float(info[4].split(' ')[1])
            half_interval = int(utterance_duration / 2.0 * self.fps)  # num frames of utterance / 2

            n_frames = raw_data.shape[0]
            mid_idx = (n_frames - 1) // 2  # video has n frames, mid point is (n-1)//2 as count starts with 0
            left_idx = random.randint(0, max(0, mid_idx - half_interval - 1))  # random.randint(a,b) chooses in [a,b]
            right_idx = random.randint(min(mid_idx + half_interval + 1, n_frames), n_frames)

            data = raw_data[left_idx:right_idx]
        else:
            data = raw_data

        preprocess_data = self.preprocessing(data)
        label = self.list[idx][1]
        return preprocess_data, label

    def __len__(self):
        return len(self._data_files)

    def collate_fn(self, batch):
        out = {}
        if len(batch) == 1:
            data, lengths, labels_np, = zip(
                *[(a, a.shape[0], b) for (a, b) in sorted(batch, key=lambda x: x[0].shape[0], reverse=True)])
            data = torch.FloatTensor(data)
            out['lengths'] = [data.size(1)]

        if len(batch) > 1:
            data_list, lengths, labels_np = zip(
                *[(a, a.shape[0], b) for (a, b) in sorted(batch, key=lambda x: x[0].shape[0], reverse=True)])
            out['lengths'] = lengths
            if data_list[0].ndim == 3:
                max_len, h, w = data_list[0].shape  # since it is sorted, the longest video is the first one
                data_np = np.zeros((len(data_list), max_len, h, w))
            elif data_list[0].ndim == 1:
                max_len = data_list[0].shape[0]
                data_np = np.zeros((len(data_list), max_len))
            for idx in range(len(data_np)):
                data_np[idx][:data_list[idx].shape[0]] = data_list[idx]
            data = torch.FloatTensor(data_np)
        out['input'] = data
        out['labels'] = torch.LongTensor(labels_np)
        return out
