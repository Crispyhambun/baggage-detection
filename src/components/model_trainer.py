
import sys
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn
import os
from tqdm import tqdm

class model_train():
    def __init__(self,train,test,valid):
        self.train = train
        self.test = test
        self.valid = valid

    def training_model():
        pass