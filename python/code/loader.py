from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from generate import CHAR_NUMBER, SEED
from PIL import Image
import torch
import os
 
BATCH_SIZE = 60
 
class ImageDataSet(Dataset):
    def __init__(self, dir_path):
        super(ImageDataSet, self).__init__()
        self.img_path_list = [f"{dir_path}/{filename}" for filename in os.listdir(dir_path)]
        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Grayscale()
        ])
 
    def __getitem__(self, idx):
        image = self.trans(Image.open(self.img_path_list[idx]))
        label = self.img_path_list[idx].split("-")[-1].replace(".png", "")
        label = one_hot_encode(label)
        return image, label
    
    def __len__(self):
        return len(self.img_path_list)
 
 
def one_hot_encode(label):
    """将字符转为独热码"""
    cols = len(SEED)
    rows = CHAR_NUMBER
    result = torch.zeros((rows, cols), dtype=float)
    for i, char in enumerate(label):
        j = SEED.index(char)
        result[i, j] = 1.0
 
    return result.view(1, -1)[0]
 
def one_hot_decode(pred_result):
    """将独热码转为字符"""
    print(pred_result.shape)
    pred_result = pred_result.view(-1, len(SEED))
    print(pred_result.shape)
    print(pred_result)
    index_list = torch.argmax(pred_result, dim=1)
    print(index_list)
    print(index_list.shape)
    text = "".join([SEED[i] for i in index_list])
    return text
 
def get_loader(path):
    """加载数据"""
    dataset = ImageDataSet(path)
    dataloader = DataLoader(dataset, BATCH_SIZE, shuffle=True)
    return dataloader
 