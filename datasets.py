import numpy as np
from PIL import Image

import csv
import json
import os

from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

from torch.utils.data import Dataset

class NormalDataset(Dataset):
    def __init__(self, root_dir, vehicle_csv, label_json, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.image_names = []
        self.labels = []

        with open(label_json, 'r') as json_file:
            data_dict = json.load(json_file)

        with open(vehicle_csv, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            header = next(csv_reader)
            for row in csv_reader:
                vehicle_id = row[0]
                for cam_id in data_dict[vehicle_id]:
                    self.image_names += [image_name for image_name in data_dict[vehicle_id][cam_id]]
                    self.labels += [int(vehicle_id) for image_name in data_dict[vehicle_id][cam_id]]
                    
    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.root_dir, self.image_names[idx]))
        label = self.labels[idx]
        if self.transform is not None:
            image = self.transform(image)
        return image, label
        
    def __len__(self):
        return len(self.image_names)

    
class TripletDataset(Dataset):
    def __init__(self, root_dir, vehicle_csv, label_json, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.image_names = []
        self.labels = []

        with open(label_json, 'r') as json_file:
            data_dict = json.load(json_file)

        with open(vehicle_csv, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            header = next(csv_reader)
            for row in csv_reader:
                vehicle_id = row[0]
                for cam_id in data_dict[vehicle_id]:
                    self.image_names += [image_name for image_name in data_dict[vehicle_id][cam_id]]
                    self.labels += [int(vehicle_id) for image_name in data_dict[vehicle_id][cam_id]]

        self.labels_set = np.asarray(list(set(self.labels)))
        self.label_to_indices = {label: np.asarray(np.where(self.labels == label)[0])
                                     for label in self.labels_set}
        self.labels_set = set(self.labels_set.tolist())


    def __getitem__(self, idx):
        image1 = Image.open(os.path.join(self.root_dir, self.image_names[idx]))
        label1 = self.labels[idx]

        pos_idx = idx
        while pos_idx == idx:
            pos_idx = np.random.choice(self.label_to_indices[label1])
            
        neg_label = np.random.choice(list(self.labels_set - set([label1])))
        neg_idx = np.random.choice(self.label_to_indices[neg_label])

        image2 = Image.open(os.path.join(self.root_dir, self.image_names[pos_idx]))
        image3 = Image.open(os.path.join(self.root_dir, self.image_names[neg_idx]))
        
#         plt.imshow(np.asarray(image1))
#         plt.show()
#         plt.imshow(np.asarray(image2))
#         plt.show()
#         plt.imshow(np.asarray(image3))
#         plt.show()

        if self.transform is not None:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
            image3 = self.transform(image3)
        return (image1, image2, image3), []

    def __len__(self):
        return len(self.image_names)
