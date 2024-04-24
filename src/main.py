import pickle
import os
import torch
import random
from torchvision.transforms import ToTensor, ToPILImage
import zipfile
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import RandomSampler, Sampler
from torchvision import transforms, utils


import os

data_folder_path = "Tema_2_DL_data"

class MNISTDataset(Dataset):
    def __init__(self, folder, filename):
        self.filename = filename
        self.data_pkl = pickle.load(open(os.path.join(folder, filename), 'rb'))
        self.images_list = self.data_pkl['images']
        self.coords_list = self.data_pkl['coords']
        self.labels_list = self.data_pkl['no_count']

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        return self.images_list[idx], self.labels_list[idx]

class_weight = None
train_dataset = None
from util import set_class_weight

def read_data():
    global train_dataset
    train_filename =  "mnist_count_train.pickle"
    test_filename = "mnist_count_test.pickle"
    # train_pkl = pickle.load(open(os.path.join(data_folder_path, train_filename),'rb'))
    # test_pkl = pickle.load(open(os.path.join(data_folder_path, test_filename), 'rb'))
    # print("There are ", len(train_pkl['images']), " in the training set")
    # print("There are ", len(test_pkl['images']), " in the training set")

    train_dataset = MNISTDataset(data_folder_path, train_filename)

    set_class_weight(train_dataset)

    sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=32, sampler=sampler)#, shuffle=True)

    labels_dict = {}
    for batch in train_dataloader:
        for data, label in list(zip(batch[0], batch[1])):
            if int(label) not in labels_dict.keys():
                labels_dict[int(label)] = 1
            else:
                labels_dict[int(label)] += 1
            # [100,100], 1,2,3,4,5
        data_batch = batch[0] # [bs, 100, 100]
        labels_batch = batch[1] # [bs]
    print("Distribution of classes in the train set", labels_dict)

    labels_dict = {}
    test_dataset = MNISTDataset(data_folder_path, test_filename)
    sampler = RandomSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, batch_size=32, sampler=sampler)#, shuffle=True)

    for batch in test_dataloader:
        for data, label in list(zip(batch[0], batch[1])):
            if int(label) not in labels_dict.keys():
                labels_dict[int(label)] = 1
            else:
                labels_dict[int(label)] += 1
            # [100,100], 1,2,3,4,5
        data_batch = batch[0] # [bs, 100, 100]
        labels_batch = batch[1] # [bs]

    print("Distribution of classes in the test set", labels_dict)

    return train_dataloader, test_dataloader

import cv2

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


from src.image_classification import task_one


use_cuda = torch.cuda.is_available()
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

import matplotlib.pyplot as plt
from src.image_localisation import task_two

from efficientnet_pytorch import EfficientNet
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchsummary
import torchvision.models as models

no_filters1 = 20
no_filter2 = 50
no_neurons1 = 500

import torch.nn.functional as F

class CNN(nn.Module):
    # the init() is called a single time, when you create the model
    # so all the layers should be created here.
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 20, kernel_size = 5, stride = 1)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(in_features = 4 * 4 * 50, out_features = 500)
        self.fc2 = nn.Linear(in_features = 500, out_features = 10)

    # the forward() is called at each iteration, so we only apply the already
    # created operations inside this function
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        if x.shape[0] == 968 and x.shape[1] == 5:
            x = x.view(-1, 968)
            x = F.relu(x)
            self.fc3 = nn.Linear(968, 32)
            x = self.fc3(x)
            x = torch.transpose(x, 1, 0)
        return F.log_softmax(x, dim=1)


def plot_all(train_loss_p, train_acc_p, test_loss_p, test_acc_p, train_loss_b, train_acc_b, test_loss_b, test_acc_b):
    print("P=Pretrained, B=Base")
    plt.plot(train_loss_p, label="loss train Pretrained")
    plt.plot(train_acc_p, label="acc train Pretrained")
    plt.plot(test_loss_p, label="loss test Pretrained")
    plt.plot(test_acc_p, label="acc test Pretrained")
    plt.plot(train_loss_b, label="loss train Base")
    plt.plot(train_acc_b, label="acc train Base")
    plt.plot(test_loss_b, label="loss test Base")
    plt.plot(test_acc_b, label="acc test Base")
    plt.legend()
    plt.show()


def main():
    train_dataloader, test_dataloader = read_data()
    global train_dataset

    tasks =  [False, True]

    if tasks[0] == True:
        network = CNN()
        learning_rate = 0.01
        momentum = 0.9
        optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                              momentum=momentum)

        network.load_state_dict(torch.load("Tema_2_DL_data/mnist_cnn_10_epochs.pt", map_location=torch.device('cpu')))
        network.eval()

        task_one(train_dataloader, test_dataloader, network)

    if tasks[1] == True:
        pretrained_model = CNN()
        learning_rate = 0.01
        momentum = 0.9
        optimizer = optim.SGD(pretrained_model.parameters(),
                              lr=learning_rate,
                              momentum=momentum)
        pretrained_model.load_state_dict(
            torch.load("Tema_2_DL_data/mnist_cnn_5_epochs.pt",
                       map_location=torch.device('cpu')))

        pretrained_model.fc1 = nn.Linear(800, 32)
        pretrained_model.fc2 = nn.Linear(32, 5)

        print("Model pretrained on MNIST:")
        print(pretrained_model)
        pretrained_model.eval()
        from util import Base_CNN

        base_model = Base_CNN()
        print(base_model)

        train_loss_b, train_acc_b, test_loss_b, test_acc_b = \
            task_two(train_dataloader, test_dataloader, data_threshold=1.0,
                     pretrained_model=pretrained_model, base_model=base_model,
                     train_dataset=train_dataset, model_type="base")

        train_loss_p, train_acc_p, test_loss_p, test_acc_p = \
            task_two(train_dataloader, test_dataloader, data_threshold=1.0,
                 pretrained_model=pretrained_model, base_model=base_model,
                 train_dataset=train_dataset, model_type="pretrained")

        plot_all(train_loss_p, train_acc_p, test_loss_p, test_acc_p,
                 train_loss_b, train_acc_b, test_loss_b, test_acc_b)

        train_loss_p, train_acc_p, test_loss_p, test_acc_p = \
            task_two(train_dataloader, test_dataloader, data_threshold=0.5,
                 pretrained_model=pretrained_model, base_model=base_model,
                 train_dataset=train_dataset, model_type="pretrained")
        train_loss_b, train_acc_b, test_loss_b, test_acc_b = \
            task_two(train_dataloader, test_dataloader, data_threshold=0.5,
                 pretrained_model=pretrained_model, base_model=base_model,
                 train_dataset=train_dataset, model_type="base")

        plot_all(train_loss_p, train_acc_p, test_loss_p, test_acc_p,
                 train_loss_b, train_acc_b, test_loss_b, test_acc_b)


        train_loss_p, train_acc_p, test_loss_p, test_acc_p = \
                task_two(train_dataloader, test_dataloader, data_threshold=0.2,
                 pretrained_model=pretrained_model, base_model=base_model,
                 train_dataset=train_dataset, model_type="pretrained")
        train_loss_b, train_acc_b, test_loss_b, test_acc_b = \
                task_two(train_dataloader, test_dataloader, data_threshold=0.2,
                 pretrained_model=pretrained_model, base_model=base_model,
                 train_dataset=train_dataset, model_type="base")

        plot_all(train_loss_p, train_acc_p, test_loss_p, test_acc_p,
                 train_loss_b, train_acc_b, test_loss_b, test_acc_b)



if __name__ =="__main__":
    main()


