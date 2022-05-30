import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

import torch
import cv2

use_cuda = torch.cuda.is_available()

def plot_loss(loss, label, color='blue'):
    plt.plot(loss, label=label, color=color)
    plt.legend()
    plt.show()


def train(args, model, device, train_loader, optimizer, epoch, data_threshold, criterion, model_type):
    global class_weight
    model.train()
    all_losses = []
    data_points_trained_on = 0
    correct = 0
    num_iter = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data_points_trained_on += len(data)
        if batch_idx > len(train_loader) * data_threshold:
            break
        # put the data on the GPU
        data, target = data.to(device), target.to(device)
        # initialize as zeros all the gradients of the model
        optimizer.zero_grad()
        num_iter += 1
        if model_type == "pretrained":
            data = torch.unsqueeze(data, 1)
            data = data.type(torch.FloatTensor)
            data /= 255.0
            # print("###")
            # print(data.shape)
            # obtain the predictions in the FORWARD pass of the network
            output = model(data)

            # print(output.shape)
            # print(target.shape)

            # compute average LOSS for the current batch
            target = target.type(torch.LongTensor)
            target -= 1
            if output.shape[0] != 32 or target.shape[0] != 32:
                break
        else:
            data = data.unsqueeze(1)
            data = data.type(torch.FloatTensor)
            data /= 255.0
            target = target.type(torch.LongTensor)
            output = model(data)
            target -= 1
        pred = output.argmax(dim=1,
                             keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).float().mean().item()

        loss = criterion(output, target)
        all_losses.append(loss.detach().cpu().numpy())
        # BACKPROPAGATE the gradients
        loss.backward()
        # use the computed gradients to OPTIMISE the model
        optimizer.step()
        # print the training loss of each batch
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
    print('The models were trained on ' + str(
        data_points_trained_on) + ' data points')

    train_accuracy = 100. * correct / num_iter
    return np.array(all_losses).mean(), train_accuracy


def test(args, model, device, test_loader, data_threshold, criterion, model_type):
    global class_weight
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        num_iter = 0
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            # obtain the prediction by a forward pass

            if model_type == "pretrained":
                # obtain the predictions in the FORWARD pass of the network
                data = torch.unsqueeze(data, 1)
                data = data.type(torch.FloatTensor)
                data /= 255.0
                # obtain the predictions in the FORWARD pass of the network
                # print('**')
                # print(data.shape)
                output = model(data)
                # print(output.shape)
                # print(target.shape)
                # compute average LOSS for the current batch
                target = target.type(torch.LongTensor)
                target -= 1
                if output.shape[0] != 32 or target.shape[0] != 32:
                    break
            else:
                data = data.unsqueeze(1)
                data = data.type(torch.FloatTensor)
                data /= 255.0
                target = target.type(torch.LongTensor)
                output = model(data)
                target -= 1

            # calculate the loss for the current batch and add it across the entire dataset
            test_loss += criterion(output, target)  # sum up batch loss
            # compute the accuracy of the predictions across the entire dataset
            # get the most probable prediction
            pred = output.argmax(dim=1,
                                 keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).float().mean().item()
            num_iter += 1
    test_loss /= num_iter
    test_accuracy = 100. * correct / num_iter
    # print the Accuracy for the entire dataset
    print('\nTest set: Average loss: {:.4f}, Accuracy: ({:.0f}%)\n'.format(
        test_loss,
        test_accuracy))
    return test_loss, test_accuracy


class_weight = []

def set_class_weight(train_dataset):
    global class_weight
    class_weight = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0])
    labels_dict = {}
    for i in range(len(train_dataset)):
        if train_dataset[i][1] not in labels_dict.keys():
            labels_dict[train_dataset[i][1]] = 1
        else:
            labels_dict[train_dataset[i][1]] += 1
    for i, key in enumerate(sorted(list(labels_dict.keys()))):
        class_weight[i] = len(train_dataset) / labels_dict[key]
    return class_weight



class Base_CNN(nn.Module):
    # the init() is called a single time, when you create the model
    # so all the layers should be created here.
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=30, kernel_size=5, stride=1)
        self.conv3 = nn.Conv2d(in_channels=30, out_channels=40, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(in_features=40 * 9 * 9, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=64)
        self.fc4 = nn.Linear(in_features=64, out_features=5)

    # the forward() is called at each iteration, so we only apply the already
    # created operations inside this function
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 40 * 9 * 9)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.1)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.1)
        x = F.relu(self.fc3(x))
        x = F.dropout(x, p=0.1)
        x = self.fc4(x)
        return x


