# -*- coding:utf-8 -*-

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from PIL import Image
import cv2
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
bs = 1
gap = 100

img_dir = '/home/jinbo/gitme/dl-algorithm-coding/data/ssd.2-200-2/JPEGImages'
anno_dir = '/home/jinbo/gitme/dl-algorithm-coding/data/ssd.2-200-2/anno'
label_dir = '/home/jinbo/gitme/dl-algorithm-coding/data/ssd.2-200-2/label'

img_names = sorted([x for x in os.listdir(img_dir)])
anno_names = sorted([x for x in os.listdir(anno_dir)])

label = []
for anno_name in anno_names:
    name = os.path.join(anno_dir, anno_name)
    #print(name)
    with open(os.path.join(anno_dir, anno_name), 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line[:2] == '10':
                with open(os.path.join(label_dir, anno_name), 'w') as fw:
                    fw.write(line[3:])
                    label.append([float(x) for x in line[3:].split()])

label = np.array(label)
label = torch.from_numpy(label).to(device).float()
#print(label.shape)
label = label.view(-1, bs, 4)
#print(label.shape)

print(label[0])

ims = []
for img_name in img_names:
    name = os.path.join(img_dir, img_name)
    im = cv2.imread(name)
    im = cv2.resize(im, (224, 224))
    im = im/255
    ims.append(im)

ims = np.array(ims)
ims = torch.from_numpy(ims).to(device).float()
#print(ims.shape)
ims = ims.view(-1, bs, 3, 224, 224)
print(type(ims))
#print(ims.shape)
#print(ims)
#print(img_names)
#for i, img ,y in enumerate(img_names[:3], anno_names[:3]):
#    print(i, img)


#print(img_names)
#print(anno_names)
#print(label)

a = torch.from_numpy(np.array([[8.,9,10,11],[2., 3, 4, 5]])).to(device)
b = torch.from_numpy(np.array([[18.,19,20,21],[3.,4,5,6]])).to(device)
print(a,b)
#for i,j in zip(a,b):
#    print(i, j)
#    print(23333)


def myloss(pred, label):
    loss = label - pred
    loss = torch.pow(loss, 2)
    loss = torch.mean(loss)
    return loss

#a = myloss(a,b)
#print(a)
a = 1

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in zip(ims[:gap//bs], label[:gap//bs]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                #print(inputs.shape)
                #print(labels.shape)
                #print('!' * 100)
                #break

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    #print(inputs.shape)
                    outputs = model(inputs)
                    #_, preds = torch.max(outputs, 1)
                    preds = outputs

                    #loss = criterion(outputs, labels)
                    loss = myloss(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                # print('size:',inputs.size(0))
                running_loss += loss.item() * inputs.size(0)
                #running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(label[:gap])
            #epoch_loss = loss.item()
            epoch_acc = 0
            #epoch_acc = running_corrects.double() / len(label)


            if epoch % 1 == 0:
                time_elapsed = time.time() - since
                print('line:153-has spend time {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
                #"""


                """
                val_inputs = ims[gap // bs + 1:]
                val_labels = label[gap // bs + 1]
                val_inputs = val_inputs.to(device)
                val_labels = val_labels.to(device)
                val_preds = model(val_inputs)
                val_loss_a = myloss(val_preds, val_labels)
                model.train(model.training)
                """
                with torch.no_grad():
                    #print(len(label))
                    #print(len(label[:gap//bs + 1]))
                    #print(len(label[gap//bs+1:]))
                    val_loss_a = 0
                    for val_inputs, val_labels in zip(ims[gap//bs + 1:], label[gap//bs + 1:]):
                        #print('ok')
                        val_inputs = val_inputs.to(device)
                        val_labels = val_labels.to(device)
                        val_preds = model(val_inputs)
                        val_loss = myloss(val_preds, val_labels)
                        model.train(model.training)
                        #print(val_loss)
                        val_loss_a = (val_loss_a + val_loss.item())/2

                #"""
                print('{} Loss: {:.4f} val_loss: {:.4f}'.format(phase, epoch_loss, val_loss_a))






            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for inputs, labels in zip(ims[gap+1:], label[gap+1:]):
            inputs = inputs.to(device)
            labels = labels.to(device)

            preds = model(inputs)
            val_loss = myloss(preds, labels)
            print('val_loss:', val_loss)
            #outputs = model(inputs)
            #_, preds = torch.max(outputs, 1)
            """
            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
            """
        model.train(mode=was_training)

model_ft = models.vgg16(pretrained=False)
#print(model_ft)
#print('*' * 25)
#print(model_ft.features[11])
model_ft.classifier[6] = nn.Linear(in_features=4096, out_features=4, bias=True)

model_ft = model_ft.to(device)

#criterion = nn.CrossEntropyLoss()
criterion = myloss

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,num_epochs=25)