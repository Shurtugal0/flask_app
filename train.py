import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
import torch.optim as optim
import time
import onnx
from datetime import datetime

device = torch.device('cuda:0')

EXPERIMENT_NAME = "MNET DEMO"
VERSION_NAME = "1.0"
EXPERIMENT_DATE = "2020-05-20 20:01:28.658895"

def train_model(model, dataloaders, criterion, optimizer, num_epochs=10):
    since = time.time()

    val_acc_history = []

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train() 
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return model, val_acc_history

def to_onnx(model):

    model.to(torch.device('cpu'))
    x = torch.randn(1, 3, 32, 32, requires_grad=True)
    torch_out = model(x)

    # Export the model
    torch.onnx.export(model,               # model being run
                      x,                         # model input (or a tuple for multiple inputs)
                      "model.onnx",   # where to save the model (can be a file or file-like object)
                      export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=10,          # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names = ['input'],   # the model's input names
                      output_names = ['output'], # the model's output names
                      dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                    'output' : {0 : 'batch_size'}})

    onnx_model = onnx.load("model.onnx")

    meta_1 = onnx_model.metadata_props.add()
    meta_2 = onnx_model.metadata_props.add()
    meta_3 = onnx_model.metadata_props.add()
    meta_1.key = "Experiment_name"
    meta_1.value = EXPERIMENT_NAME
    meta_2.key = "Version"
    meta_2.value = VERSION_NAME
    meta_3.key = "Date"
    meta_3.value = EXPERIMENT_DATE
    onnx.save(onnx_model, './model.onnx')


if __name__ == '__main__':
    train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomResizedCrop((32, 32), scale=(0.7, 1.0)),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    val_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=500,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=val_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=500,
                                             shuffle=False, num_workers=2)

    model = torchvision.models.mobilenet_v3_small(pretrained=True)
    model.classifier[-1] = nn.Linear(1024, 10)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model, history = train_model(model, {'train': trainloader, 'val': testloader}, 
                                                            criterion, optimizer)

    to_onnx(model)

