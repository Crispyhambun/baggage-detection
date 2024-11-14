import os
import torch
import sys
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.logger import logging
from src.exception import Custom_Exception

class DataIngestionConfig:
    data_dir = "C:/Users/octav/Downloads/xray_images"
    train_images_dir = os.path.join(data_dir, "train", "images")
    train_labels_dir = os.path.join(data_dir, "train", "labels")
    
    test_images_dir = os.path.join(data_dir, "test", "images")
    test_labels_dir = os.path.join(data_dir, "test", "labels")
    
    valid_images_dir = os.path.join(data_dir, "valid", "images")
    valid_labels_dir = os.path.join(data_dir, "valid", "labels")

class CustomDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
       
        self.image_files = sorted(os.listdir(images_dir))
        self.label_files = sorted(os.listdir(labels_dir))
        
        

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.images_dir, self.image_files[idx])
        label_path = os.path.join(self.labels_dir, self.label_files[idx])


        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Load bounding box and class label
        boxes = []
        labels = []
        with open(label_path, "r") as file:
            for line in file:
                parts = line.strip().split()
                class_id = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:])
                boxes.append([x_center, y_center, width, height])
                labels.append(class_id)

        # Convert lists to tensors
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        return image, {"boxes": boxes, "labels": labels}

class Dataingestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def initiate_data_ingestion(self):
        try:
            logging.info("started Data ingestion")
            train_dataset = CustomDataset(
                images_dir=self.ingestion_config.train_images_dir,
                labels_dir=self.ingestion_config.train_labels_dir,
                transform=self.transform
            )
            valid_dataset = CustomDataset(
                images_dir=self.ingestion_config.valid_images_dir,
                labels_dir=self.ingestion_config.valid_labels_dir,
                transform=self.transform
            )
            test_dataset = CustomDataset(
                images_dir=self.ingestion_config.test_images_dir,
                labels_dir=self.ingestion_config.test_labels_dir,
                transform=self.transform
            )

            logging.info("Loading the data")

            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
            logging.info("Loading Successful")

            return train_loader, valid_loader, test_loader

        except Exception as e:
            raise Custom_Exception(e, sys)

# Example usage
data_ingestion = Dataingestion()
train_loader, valid_loader, test_loader = data_ingestion.initiate_data_ingestion()
