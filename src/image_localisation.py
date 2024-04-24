import matplotlib.pyplot as plt
from util import set_class_weight, Base_CNN, train, test, plot_loss
import torch.nn as nn
import torch.optim as optim
import torch

kwargs={}
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

class Args():
  def __init__(self):
      self.batch_size = 64
      self.test_batch_size = 64
      self.epochs = 10
      self.lr = 0.01
      self.momentum = 0.9
      self.seed = 1
      self.log_interval = int(10000 / self.batch_size)
      self.cuda = False

args = Args()



def task_two(train_dataloader, test_dataloader, data_threshold, base_model, pretrained_model, train_dataset, model_type):
    class_weight = set_class_weight(train_dataset)
    if model_type == "pretrained":
        model_chosen = pretrained_model
    else:
        model_chosen = base_model
    for model in [model_chosen]:
        print("*"*19)
        if model == base_model:
            print("Model with random initialized weights")
        else:
            print("Model pretrained on MNIST:")
        data_points_trained_on = 0
        num_epochs = 8

        losses_train = []
        losses_test = []
        accuracy_test = []
        accuracy_train = []

        for epoch in range(0, num_epochs):
            if model == pretrained_model:
                loss = nn.NLLLoss()
                optimizer = optim.Adam(base_model.parameters(), lr=1e-3)
                train_loss, train_accuracy = train(args, model, device,
                                                   train_dataloader, optimizer,
                                                   epoch, data_threshold, loss ,
                                                   "pretrained")
                test_loss, test_accuracy = test(args, model, device,
                                                test_dataloader, data_threshold,
                                                loss, "pretrained")
            else:
                optimizer = optim.Adam(base_model.parameters(), lr=1e-3)
                loss = nn.CrossEntropyLoss(weight=class_weight)
                train_loss, train_accuracy = train(args, model, device,
                                                   train_dataloader, optimizer,
                               epoch, data_threshold, loss, "base")
                test_loss, test_accuracy = test(args, model, device, test_dataloader,
                                    data_threshold, loss, "base")

            losses_train.append(train_loss)
            losses_test.append(test_loss)

            accuracy_test.append(test_accuracy)
            accuracy_train.append(train_accuracy)
        # plot the loss/accuracy
        plt.figure(1)
        plot_loss(losses_train, 'train_loss', 'red')
        plot_loss(losses_test, 'test_loss')
        plt.figure(2)
        plot_loss(accuracy_test, 'test_accuracy')
        plot_loss(accuracy_train, 'train_accuracy')
    return losses_train, accuracy_train, accuracy_test, train_accuracy
    # save the final model
    # torch.save(base_model.state_dict(), "mnist_cnn_" + str(data_threshold) + "_.pt")
