import os
import torch
from PIL import Image

from torch.utils.data import Dataset


class ImagesDataset(Dataset):
    """Images dataset for causal metrics benchmark"""

    def __init__(self, txt_file, labels_file, root_dir, transform=None):
        """
        Args:
            txt_file (string): Path to the txt listing files
            root_dir (string): Directory with the images inside
            labels_file (string): Path to txt with labels
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.file_path = []
        self.file_name = []
        self.labels = {}
        labels = open(labels_file)
        files = open(txt_file)
        for file in files:
            self.file_path.append(os.path.join(root_dir, file.rstrip()))
            self.file_name.append(file.rstrip())

        for label in labels:
            name, id, _ = label.rstrip().split(' ')
            self.labels[name] = id

        self.transform = transform

    def __len__(self):
        return len(self.file_path)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()


        image = Image.open(self.file_path[idx]).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, self.labels[self.file_name[idx]], self.file_name[idx]
