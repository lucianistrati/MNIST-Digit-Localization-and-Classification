import torch
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


def task_one(train_dataloader, test_dataloader, pretrained_model):
    train_pred = []
    train_true = []

    val_pred = []
    val_true = []

    test_pred = []
    test_true = []

    SW_SIZE = 28
    CONF_THRESHOLD = 0.9

    STEP = 3
    BATCH_SIZE = 1
    subimages_batch = []
    img_cnt = 0
    confidences = []
    for batch in train_dataloader:
        for data, label in list(zip(batch[0], batch[1])):
            img_cnt += 1
            if img_cnt % 500 == 0:
                print(img_cnt)
            image = data
            new_image = image.type(torch.FloatTensor)
            for i in range(new_image.shape[0]):
                for j in range(new_image.shape[1]):
                    new_image[i,j] /= 255.0
            image = new_image
            label = int(label)

            train_true.append(label)

            from copy import deepcopy
            predicted_digits_set = set()

            for i in range(0, image.shape[0] - SW_SIZE, STEP):
                for j in range(0, image.shape[1] - SW_SIZE, STEP):
                    subimage = image[i:i + SW_SIZE, j:j + SW_SIZE]
                    if len(subimages_batch) < BATCH_SIZE:
                        subimage = subimage.cpu().numpy()
                        subimage_plot = cv2.cvtColor(subimage, cv2.COLOR_GRAY2RGB)
                        subimage = torch.Tensor(subimage)
                        subimages_batch.append(subimage)
                    else:
                        subimages_batch = torch.stack(subimages_batch)

                        subimages_batch = torch.unsqueeze(subimages_batch, 0)

                        output = pretrained_model(subimages_batch)
                        predicted_label = int(output.argmax(axis=1))
                        softmaxed_arr = torch.softmax(output, dim=-1)
                        conf = float(softmaxed_arr[0, predicted_label])
                        #print(conf)
                        confidences.append(conf)

                        if conf >= CONF_THRESHOLD:
                            predicted_digits_set.add(predicted_label)
                        subimages_batch = []

            if len(predicted_digits_set) == 0:
                train_pred.append(1)
            elif 1 <= len(predicted_digits_set) <= 5:
                train_pred.append(len(predicted_digits_set))
            else:
                train_pred.append(5)


    plt.hist(confidences)
    plt.show()
    print("STEP: ", STEP)
    print("Used confidence threshold: ", CONF_THRESHOLD)
    print("Accuracy score(train set):\n", accuracy_score(train_true, train_pred))
    print("Confusion matrix(train set):\n", confusion_matrix(train_true, train_pred))
    print("Classif report(train set):\n", classification_report(train_true, train_pred))