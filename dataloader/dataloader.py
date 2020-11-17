import os, glob, random
from PIL import Image

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class ImageDataset(Dataset):
    def __init__(self, data_name="selfie2anime", mode="train"):
        super().__init__()
        self.mode = mode
        self.data_path = "dataloader/data/%s/" % data_name
        self.files_A = sorted(glob.glob(os.path.join(self.data_path + self.mode + "/A/*.*")))
        self.files_B = sorted(glob.glob(os.path.join(self.data_path + self.mode + "/B/*.*")))
        self.transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor()])


    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

    def __getitem__(self, idx):
        img_A = self.transform(Image.open(self.files_A[idx % len(self.files_A)]).convert("RGB"))
        img_B = self.transform(Image.open(self.files_B[idx % len(self.files_B)]).convert("RGB"))
        # real_A = self.transform(Image.open(random.choice(self.files_A)).convert("RGB"))
        # real_B = self.transform(Image.open(random.choice(self.files_B)).convert("RGB"))
        return img_A, img_B#, real_A, real_B


def Loader(data_name="selfie2anime", mode="train", batch_size=1, num_workers=2):
    dataset = ImageDataset(data_name, mode)
    loader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        shuffle=True)
    return loader

