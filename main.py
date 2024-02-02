"""import splitfolders

splitfolders.ratio('RPS', output="data", ratio=(0.7, 0.15, 0.15))"""

from timeit import default_timer as timer
import numpy as np
from torchvision import datasets
from torchvision.transforms import v2 as transform
from tqdm import tqdm
from CNN_Classes import *
from functions import *

device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(42)
torch.cuda.manual_seed(42)

model = RPSModel4(num_channels=1).to(device)

means, stds = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

train_transform = transform.Compose([
    transform.ToTensor(),
    transform.Normalize(means, stds),
    transform.RandomAffine(0, shear=0.2),  # random shear 0.2
    transform.RandomAffine(0, scale=(0.8, 1.2)),  # random zoom 0.2
    transform.RandomRotation(20),
    transform.RandomHorizontalFlip(),
    transform.RandomVerticalFlip(),
])

test_transform = transform.Compose([
    transform.ToTensor(),
    transform.Normalize(means, stds),
])

train_transform_gray = transform.Compose([
    transform.ToTensor(),
    transform.Normalize(means, stds),
    transform.RandomAffine(0, shear=0.2),  # random shear 0.2
    transform.RandomAffine(0, scale=(0.8, 1.2)),  # random zoom 0.2
    transform.RandomRotation(20),
    transform.RandomHorizontalFlip(),
    transform.RandomVerticalFlip(),
    transform.Grayscale(),
    # transform.CenterCrop((224, 224)),
])

test_transform_gray = transform.Compose([
    transform.ToTensor(),
    transform.Normalize(means, stds),
    transform.Grayscale(),
    # transform.Resize((128, 128)),
    # transform.CenterCrop((128, 128)),
])

train_set = datasets.ImageFolder(root='data/train', transform=train_transform_gray)
validation_set = datasets.ImageFolder(root='data/val', transform=test_transform_gray)
test_set = datasets.ImageFolder(root='data/test', transform=test_transform_gray)

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

torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Setup loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.005)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

torch.manual_seed(42)
torch.cuda.manual_seed(42)

train_time_start_model_2 = timer()

# Train and test model
train_loss_array = np.array([])
train_acc_array = np.array([])

validation_loss_array = np.array([])
validation_acc_array = np.array([])

epochs = 25
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
validation_results = eval_model(
    model=model,
    data_loader=validation_loader,
    loss_fn=loss_fn,
    accuracy_fn=accuracy_fn
)
test_results = eval_model(
    model=model,
    data_loader=test_loader,
    loss_fn=loss_fn,
    accuracy_fn=accuracy_fn
)
print(f" Validation data prediction result is: {validation_results}")
print(f" Test data prediction result is: {test_results}")
batch = next(iter(validation_loader))
n_images = 5
images = batch[0][:n_images, ...].cuda()

model.net[0].register_forward_hook(get_activation('conv1'))
model.net[5].register_forward_hook(get_activation('conv2'))
model.net[10].register_forward_hook(get_activation('conv3'))

if "RPSModel2" in str(type(model)):
    model.net[15].register_forward_hook(get_activation('conv4'))

if "RPSModel3" in str(type(model)):
    model.net[15].register_forward_hook(get_activation('conv4'))
    model.net[20].register_forward_hook(get_activation('conv5'))

if "RPSModel4" in str(type(model)):
    model.net[15].register_forward_hook(get_activation('conv4'))
    model.net[20].register_forward_hook(get_activation('conv5'))
    model.net[25].register_forward_hook(get_activation('conv6'))

new_image = torch.unsqueeze(images[0], 0)
output = model(new_image)

act1 = activation['conv1'].squeeze().cpu()
act2 = activation['conv2'].squeeze().cpu()
act3 = activation['conv3'].squeeze().cpu()

visualize_feature_map(images[0].permute(1, 2, 0).cpu().numpy(), act1)
visualize_feature_map(images[0].permute(1, 2, 0).cpu().numpy(), act2)
visualize_feature_map(images[0].permute(1, 2, 0).cpu().numpy(), act3)

if "RPSModel2" in str(type(model)):
    act4 = activation['conv4'].squeeze().cpu()
    visualize_feature_map(images[0].permute(1, 2, 0).cpu().numpy(), act4)

if "RPSModel3" in str(type(model)):
    act4 = activation['conv4'].squeeze().cpu()
    act5 = activation['conv5'].squeeze().cpu()
    visualize_feature_map(images[0].permute(1, 2, 0).cpu().numpy(), act4)
    visualize_feature_map(images[0].permute(1, 2, 0).cpu().numpy(), act5)

if "RPSModel4" in str(type(model)):
    act4 = activation['conv4'].squeeze().cpu()
    act5 = activation['conv5'].squeeze().cpu()
    act6 = activation['conv6'].squeeze().cpu()
    visualize_feature_map(images[0].permute(1, 2, 0).cpu().numpy(), act4)
    visualize_feature_map(images[0].permute(1, 2, 0).cpu().numpy(), act5)
    visualize_feature_map(images[0].permute(1, 2, 0).cpu().numpy(), act6)
