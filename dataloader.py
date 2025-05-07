"""
Dataloader for dataset
/data/train(split 80% train, 20% valid)
/test_release
"""
import os
import json
import torch
import numpy as np
import cv2
from skimage import io
from torch.utils.data import Dataset, DataLoader, random_split


class TrainDataset(Dataset):
    """
    Dataset for training and validation
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir # train
        self.transform = transform
        # folder
        self.image_folders = [f for f in os.listdir(root_dir)
                              if os.path.isdir(os.path.join(root_dir, f))]
        self.image_folders.sort()

    def __len__(self):
        return len(self.image_folders)

    def __getitem__(self, index):
        folder = os.path.join(self.root_dir, self.image_folders[index])
        image_file = [f for f in os.listdir(folder) if f.startswith('image')][0]
        image_path = os.path.join(folder, image_file)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        masks = []
        boxes = []
        labels = []
        for class_idx in range(1, 5):
            mask_path = os.path.join(folder, f'class{class_idx}.tif')

            if os.path.exists(mask_path):
                mask = io.imread(mask_path)

                unique_labels = np.unique(mask)
                for label in unique_labels:
                    if label == 0:
                        continue
                    binary_mask = np.zeros_like(mask, dtype=np.uint8)
                    binary_mask[mask == label] = 1
                    pos = np.where(binary_mask == 1)
                    if len(pos[0]) == 0 or len(pos[1]) == 0:
                        continue
                    y1, x1 = np.min(pos[0]), np.min(pos[1])
                    y2, x2 = np.max(pos[0]), np.max(pos[1])

                    masks.append(binary_mask)
                    boxes.append([x1, y1, x2, y2])
                    labels.append(class_idx)

        if self.transform is not None:
            image = self.transform(image)

        target = {}

        if len(boxes) > 0:
            target['boxes'] = torch.tensor(boxes, dtype=torch.float32)
            target['labels'] = torch.tensor(labels, dtype=torch.int64)

            masks_array = np.stack(masks, axis=0)
            target['masks'] = torch.from_numpy(masks_array).to(torch.uint8)
            target['image_id'] = torch.tensor([index])

            area = [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes]
            target['area'] = torch.tensor(area, dtype=torch.float32)
            target['iscrowd'] = torch.zeros((len(boxes),), dtype=torch.int64)
        else:
            h, w = image.shape[1], image.shape[2]
            target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
            target['labels'] = torch.zeros((0,), dtype=torch.int64)
            target['masks'] = torch.zeros((0, h, w), dtype=torch.uint8)
            target['image_id'] = torch.tensor([index])
            target['area'] = torch.zeros((0,), dtype=torch.float32)
            target['iscrowd'] = torch.zeros((0,), dtype=torch.int64)

        return image, target

class TestDataset(Dataset):
    """
    Dataset for test
    """
    def __init__(self, root_dir, json_file, transform=None):
        self.transform = transform
        self.test_folder = os.path.join(root_dir, 'test_release')
        self.json_file_path = os.path.join(root_dir, json_file)
        with open(self.json_file_path, 'r') as f:
            self.name_to_ids = json.load(f)

        self.name_to_ids.sort(key=lambda x: x["id"])

    def __len__(self):
        return len(self.name_to_ids)

    def __getitem__(self, idx):
        info = self.name_to_ids[idx]
        file_name = info["file_name"]
        image_id = info["id"]
        height = info["height"]
        width = info["width"]
        image_path = os.path.join(self.test_folder, file_name)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            image = self.transform(image)

        return image, image_id, height, width


class Transform:
    """
    Transform
    ImageNet normalization
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        image = image.astype(np.float32) / 255.0

        for i in range(3):
            image[:, :, i] = (image[:, :, i] - self.mean[i]) / self.std[i]

        image = np.transpose(image, (2, 0, 1))
        image = torch.from_numpy(image).float()

        return image

class GetLoader:
    """
    Get dataset
    train_loader, valid_loader, test_loader
    """
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        self.batch_size = 1
        self.num_workers = 4

        self.transform = Transform(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def collate_fn(self, batch):
        return tuple(zip(*batch))

    def train_loader(self):
        """
        Return train loader
        """
        train_dataset = self.split_dataset()[0]

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

        return train_loader, train_dataset

    def valid_loader(self):
        """
        Return valid loader
        """
        val_dataset = self.split_dataset()[1]

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

        return val_loader, val_dataset

    def test_loader(self):
        """
        Return test loader
        """
        test_dataset = TestDataset(
            root_dir=self.data_dir,
            json_file='test_image_name_to_ids.json',
            transform=self.transform,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

        return test_loader, test_dataset

    def split_dataset(self):
        """
        split dataset into train and valid
        80% train, 20% valid
        """
        dataset = TrainDataset(
            root_dir=os.path.join(self.data_dir, 'train'),
            transform=self.transform,
        )

        dataset_size = len(dataset)
        val_size = int(dataset_size * 0.2)
        train_size = dataset_size - val_size
        generator = torch.Generator().manual_seed(42)
        train_dataset, val_dataset = random_split(
            dataset,
            [train_size, val_size],
            generator=generator
        )

        return train_dataset, val_dataset
