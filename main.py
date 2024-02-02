"""import splitfolders

splitfolders.ratio('RPS', output="data", ratio=(0.7, 0.15, 0.15))"""

import torch
from torch import nn
from tqdm import tqdm
import numpy as np
from sklearn import manifold
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.transforms import v2 as transform
import splitfolders
from torch.utils.data import DataLoader, Dataset
import torch
import requests
from pathlib import Path
from helper_functions import accuracy_fn
from timeit import default_timer as timer
from CNN_Classes import *


def normalize_image(image):
    image_min = image.min()
    image_max = image.max()
    image.clamp_(min=image_min, max=image_max)
    image.add_(-image_min).div_(image_max - image_min + 1e-5)
    return image


device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(42)
torch.cuda.manual_seed(42)

model = RPSModel1(num_channels=3).to(device)

means, stds = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

train_transform = transform.Compose([
    transform.ToTensor(),
    transform.Normalize(means, stds),
    transform.RandomAffine(0, shear=0.2),  # random shear 0.2
    transform.RandomAffine(0, scale=(0.8, 1.2)),  # random zoom 0.2
    transform.RandomRotation(20),
    transform.RandomHorizontalFlip(),
    transform.RandomVerticalFlip(),
    # transform.CenterCrop((224, 224)),
])

test_transform = transform.Compose([
    transform.ToTensor(),
    transform.Normalize(means, stds)
    # transform.Resize((128, 128)),
    # transform.CenterCrop((128, 128)),
])

train_set = datasets.ImageFolder(root='data/train', transform=train_transform)
validation_set = datasets.ImageFolder(root='data/val', transform=test_transform)
test_set = datasets.ImageFolder(root='data/test', transform=test_transform)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
validation_loader = DataLoader(validation_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

LABELS = ['Paper', 'Rock', 'Scissors']
data_iter = iter(train_loader)
sample_images, sample_labels = next(data_iter)
sample_images, sample_labels = sample_images.cpu().numpy(), sample_labels.cpu().numpy()

plt.figure(figsize=(20, 6))
for i in range(64):
    plt.subplot(4, 16, i + 1)
    image = sample_images[i]
    label_text = LABELS[sample_labels[i]]
    image = image.transpose((1, 2, 0))
    plt.imshow(image.squeeze(), cmap='gray')
    plt.title(label_text)
    plt.axis('off')
plt.suptitle("Sample Images from Train Dataset")
plt.show()
plt.close()

class_names = train_set.classes


def print_train_time(start: float, end: float, device: torch.device = None):
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time


def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device = device):
    train_loss, train_acc = 0, 0
    model.to(device)
    for batch, (X, y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss
        train_acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Calculate loss and accuracy per epoch and print out what's happening
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")
    return train_loss, train_acc


def test_step(data_loader: torch.utils.data.DataLoader,
              model: torch.nn.Module,
              loss_fn: torch.nn.Module,
              accuracy_fn,
              device: torch.device = device):
    test_loss, test_acc = 0, 0
    model.to(device)
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            test_pred = model(X)
            test_loss += loss_fn(test_pred, y)
            test_acc += accuracy_fn(y_true=y, y_pred=test_pred.argmax(dim=1))
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")
    return test_loss, test_acc


torch.manual_seed(42)


def eval_model(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               accuracy_fn,
               device: torch.device = device):
    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            # Send data to the target device
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))

        # Scale loss and acc
        loss /= len(data_loader)
        acc /= len(data_loader)
    return {"model_name": model.__class__.__name__,  # only works when model was created with a class
            "model_loss": loss.item(),
            "model_acc": acc}


# Setup loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

torch.manual_seed(42)
torch.cuda.manual_seed(42)

train_time_start_model_2 = timer()

# Train and test model
train_loss_array = np.array([])
train_acc_array = np.array([])

validation_loss_array = np.array([])
validation_acc_array = np.array([])

epochs = 1
for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n---------")

    train_loss, train_acc = train_step(data_loader=train_loader,
                                       model=model,
                                       loss_fn=loss_fn,
                                       optimizer=optimizer,
                                       accuracy_fn=accuracy_fn,
                                       device=device
                                       )
    validation_loss, validation_acc = test_step(data_loader=validation_loader,
                                                model=model,
                                                loss_fn=loss_fn,
                                                accuracy_fn=accuracy_fn,
                                                device=device
                                                )

    train_loss_array = np.append(train_loss_array, train_loss.cpu().detach().numpy())
    validation_loss_array = np.append(validation_loss_array, validation_loss.cpu())

    train_acc_array = np.append(train_acc_array, train_acc)
    validation_acc_array = np.append(validation_acc_array, validation_acc)

train_time_end_model_2 = timer()
total_train_time_model_2 = print_train_time(start=train_time_start_model_2,
                                            end=train_time_end_model_2,
                                            device=device)

plt.plot(range(len(train_loss_array)), train_loss_array, label='train loss')
plt.plot(range(len(validation_loss_array)), validation_loss_array, label='validation loss')
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend()
plt.show()
print("######################################################################")
plt.plot(range(len(train_acc_array)), train_acc_array, label='train accuracy')
plt.plot(range(len(validation_acc_array)), validation_acc_array, label='validation accuracy')
plt.ylabel("Accuracy")
plt.xlabel("Epochs")
plt.legend()
plt.show()

# Get model_2 results
model_2_results = eval_model(
    model=model,
    data_loader=validation_loader,
    loss_fn=loss_fn,
    accuracy_fn=accuracy_fn
)

batch = next(iter(validation_loader))
n_images = 5
images = batch[0][:n_images, ...].cuda()

n_filters = 7

for name, module in model.net.named_children():
    print(f"name is: {name}")
    print(f"module is: {str(module)}")
    """if "Conv" in str(module):
        filters = model.net[int(name)].weight.data
        filters = filters[:n_filters]

        filtered_images = F.conv2d(images, filters)

        fig = plt.figure(figsize=(15, 15))

        for i in range(n_images):

            image = images[i]

            image = normalize_image(image)

            ax = fig.add_subplot(n_images, n_filters + 1, i + 1 + (i * n_filters))
            ax.imshow(image.permute(1, 2, 0).cpu().numpy())
            ax.set_title('Original')
            ax.axis('off')

            for j in range(n_filters):
                image = filtered_images[i][j]

                image = normalize_image(image)

                ax = fig.add_subplot(n_images, n_filters + 1, i + 1 + (i * n_filters) + j + 1)
                ax.imshow(image.cpu().numpy(), cmap='bone')
                ax.set_title(f'Filter {j + 1}')
                ax.axis('off')

        fig.subplots_adjust(hspace=-0.7)
        plt.title("Output for Convolution Layer"+str(int(name)+1))
        plt.show()"""

activation = {}


def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()

    return hook


model.net[0].register_forward_hook(get_activation('conv1'))
new_image = torch.unsqueeze(images[0], 0)
output = model(new_image)

act = activation['conv1'].squeeze().cpu()

n_filters = 7
fig = plt.figure(figsize=(15, 2))
ax = fig.add_subplot(n_images, n_filters + 1, 2 + n_filters)
ax.imshow(images[0].permute(1, 2, 0).cpu().numpy())
ax.set_title('Original')
ax.axis('off')
for j in range(n_filters):
    image = act[j]
    ax = fig.add_subplot(n_images, n_filters + 1, 2 + n_filters + j + 1)
    ax.imshow(image.cpu().numpy(), cmap='bone')
    ax.set_title(f'Filter {j + 1}')
    ax.axis('off')

fig.subplots_adjust(hspace=-0.7)
plt.show()

filters = model.net[0].weight.data
filters = filters[:n_filters]

filtered_images = F.conv2d(images, filters)

fig = plt.figure(figsize=(15, 15))

for i in range(n_images):

    image = images[i]

    image = normalize_image(image)

    ax = fig.add_subplot(n_images, n_filters + 1, i + 1 + (i * n_filters))
    ax.imshow(image.permute(1, 2, 0).cpu().numpy())
    ax.set_title('Original')
    ax.axis('off')

    for j in range(n_filters):
        image = filtered_images[i][j]

        image = normalize_image(image)

        ax = fig.add_subplot(n_images, n_filters + 1, i + 1 + (i * n_filters) + j + 1)
        ax.imshow(image.cpu().numpy(), cmap='bone')
        ax.set_title(f'Filter {j + 1}')
        ax.axis('off')

fig.subplots_adjust(hspace=-0.7)
plt.show()

filters = model.net[0].weight.data
filters = filters.cpu()

n_filters = filters.shape[0]

rows = int(np.sqrt(n_filters))
cols = int(np.sqrt(n_filters))

fig = plt.figure(figsize=(30, 15))

for i in range(rows * cols):
    image = filters[i]

    image = normalize_image(image)

    ax = fig.add_subplot(rows, cols, i + 1)
    ax.imshow(image.permute(1, 2, 0))
    ax.axis('off')

fig.subplots_adjust(wspace=-0.9)
plt.show()


def make_predictions_2(model: torch.nn.Module, data: list, device: torch.device = device):
    pred_probs = []
    model.eval()
    with torch.inference_mode():
        for sample in data:
            # Prepare sample
            sample = torch.unsqueeze(sample, dim=0).to(device)  # Add an extra dimension and send sample to device

            # Forward pass (model outputs raw logit)
            pred_logit = model(sample)

            # Get prediction probability (logit -> prediction probability)
            pred_prob = torch.softmax(pred_logit.squeeze(),
                                      dim=0)  # note: perform softmax on the "logits" dimension, not "batch" dimension (in this case we have a batch size of 1, so can perform on dim=0)

            # Get pred_prob off GPU for further calculations
            pred_probs.append(pred_prob.cpu())

    # Stack the pred_probs to turn list into a tensor
    return torch.stack(pred_probs)


def make_predictions(model: torch.nn.Module, data_loader, device: torch.device = device):
    pred_probs = []
    y_true = []
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            # Send data to the target device
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            y_pred = y_pred.argmax(dim=1)
            pred_probs.append(y_pred.cpu().detach().numpy()[0])
            y_true.append(y.cpu().detach().numpy()[0])

    # Stack the pred_probs to turn list into a tensor
    return pred_probs, y_true


import random

random.seed(42)
test_samples = []
test_labels = []
for sample, label in random.sample(list(validation_set), k=9):
    test_samples.append(sample)
    test_labels.append(label)

# View the first test sample shape and label
print(
    f"Test sample image shape: {test_samples[0].shape}\nTest sample label: {test_labels[0]} ({class_names[test_labels[0]]})")

y_pred, y_true = make_predictions(model, test_loader)
y_pred = np.array(y_pred)
y_true = np.array(y_true)
correct_preds = (y_pred == y_true).sum()
pred_acc = correct_preds / len(y_true)
print(f"We got {correct_preds} predictions of {len(y_true)} samples correct! Test acc: {pred_acc * 100.0:.3f}%")

rand_indexes = np.random.randint(0, len(y_true), 64)
"""print(f"Actuals (50 random samples)    : {y_true[rand_indexes]}")
print(f"Predictions (50 random samples): {y_pred[rand_indexes]}")"""

"""NUM_SAMPLES = 64
testloader = torch.utils.data.DataLoader(test_set, batch_size=NUM_SAMPLES, shuffle=False)
data_iter = iter(testloader)
sample_images, sample_labels = next(data_iter)
sample_images, sample_labels = sample_images.cpu().numpy(), sample_labels.cpu().numpy()
predictions = y_pred[:NUM_SAMPLES]
print(f"images.shape: {sample_images.shape} - labels.shape: {sample_labels.shape}")

plt.figure(figsize=(16, 6))
for i in range(NUM_SAMPLES):
    plt.subplot(4, 16, i + 1)
    image = sample_images[i]
    label_text = class_names[sample_labels[i]]
    image = image.transpose((1, 2, 0))
    plt.imshow(image, cmap='gray')
    title = plt.title(label_text)
    plt.setp(title, color=("g" if sample_labels[i] == y_pred[i] else "r"))
    plt.axis('off')
plt.suptitle("Sample Predictions")
plt.show()
plt.close()"""
