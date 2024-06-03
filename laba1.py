import gc
import random

import numpy as np
import seaborn as sns
import torch
import torchvision.models as models
from matplotlib import pyplot as plt
from matplotlib import rcParams
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import Compose, RandomRotation, \
    RandomHorizontalFlip
from tqdm import tqdm

rcParams['figure.figsize'] = (15, 4)
sns.set(style="darkgrid", font_scale=1.4)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
GPU = torch.device('cuda:0')
CPU = torch.device('cpu')
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')

torch.cuda.empty_cache()
gc.collect()

if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), 'GB')
else:
    print('CUDA is not available. Training on CPU ...')

X = np.load('input/x_train.npy')
Y = np.load('input/y_train.npy')

print(X.shape)
print(Y.shape)
print(Y[0])

plt.figure(figsize=(18, 6))
for i in range(6):
    j = 100 + 10 * i
    plt.subplot(2, 6, i + 1)
    plt.axis("off")
    plt.imshow(X[j])

    plt.subplot(2, 6, i + 7)
    plt.axis("off")
    plt.imshow(Y[j].squeeze(), cmap='gray')
plt.show()

unique, counts = np.unique(Y, return_counts=True)
result = np.column_stack((unique, counts))
print(result)

sum_by_image = np.sum(Y, (1, 2, 3))
print(sum_by_image.shape)
unique, counts = np.unique(sum_by_image, return_counts=True)
result = np.column_stack((unique, counts))
print(result)

threshold = 200
Y_c = np.array(sum_by_image > threshold, dtype='uint8')
print(Y_c.shape)

print(Y_c[0:100])

plt.figure(figsize=(18, 6))
for i in range(6):
    j = 2 + i
    plt.subplot(2, 6, i + 1)
    plt.axis("off")
    plt.imshow(X[j])

    plt.subplot(2, 6, i + 7)
    plt.axis("off")
    plt.imshow(Y[j].squeeze(), cmap='gray')
plt.show()

X_r = np.transpose(X, axes=(0, 3, 1, 2))
Y_r = Y_c
print(X_r.shape, Y_r.shape)

X_f = X_r.astype(np.float32) / 255.0
X_t = torch.FloatTensor(X_f)

for x in X_t:
    transforms.functional.normalize(x, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True)

Y_t = torch.FloatTensor(Y_r)

ix = np.random.choice(len(X_t), len(X_t), False)
val_index = 2500
test_index = 2800
tr, val, ts = np.split(ix, [val_index, test_index])

X_train_t = X_t[tr]
X_val_t = X_t[val]
X_test_t = X_t[ts]

Y_train_t = Y_t[tr]
Y_val_t = Y_t[val]
Y_test_t = Y_t[ts]

train_dataset = TensorDataset(X_train_t, Y_train_t)
val_dataset = TensorDataset(X_val_t, Y_val_t)
test_dataset = TensorDataset(X_test_t, Y_test_t)

unique, counts = np.unique(Y_train_t, return_counts=True)
result = np.column_stack((unique, counts))
print(result)


def imshow(inp, title=None, plt_ax=plt):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt_ax.imshow(inp)
    if title is not None:
        if hasattr(plt_ax, 'set_title'):
            plt_ax.set_title(title)
        if hasattr(plt_ax, 'title'):
            plt_ax.title(title)
    plt_ax.grid(False)


imshow(X_train_t[0])

unique, counts = np.unique(Y_r, return_counts=True)
result = np.column_stack((unique, counts))
print(result)

unique, counts = np.unique(Y_r[tr], return_counts=True)
w = np.min(counts) / counts
print(w)

weights = [w[0] if x == 0 else w[1] for x in Y_r[tr]]
print(np.array(weights).shape)

print(weights)

weights = [w[0] if x == 0 else w[1] for x in Y_r[tr]]
sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights),
                                                 replacement=True)

print(Y_r[tr][50:100:5])
print(weights[50:100:5])
print(np.array(weights).shape)
print(np.array(weights[50:100:5]).shape)

train_dataloader = DataLoader(train_dataset, batch_size=128, sampler=sampler)
val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

YY_check = np.array([])

for _, y in train_dataloader:
    YY_check = np.hstack((YY_check, y.detach().numpy()))

unique, counts = np.unique(YY_check, return_counts=True)
result = np.column_stack((unique, counts))
print(result)

from sklearn.metrics import balanced_accuracy_score


class SimpleCnn(nn.Module):

    def __init__(self, size, n_classes):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(8)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(16)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(32)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(64)
        )

        fc_inputs = int((size / 16) * (size / 16) * 64)

        self.out = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fc_inputs, 96 * 5),
            nn.ReLU(),
            nn.BatchNorm1d(96 * 5),
            nn.Dropout(p=0.2),
            nn.Linear(96 * 5, 96 * 4),
            nn.ReLU(),
            nn.BatchNorm1d(96 * 4),
            nn.Dropout(p=0.1),
            nn.Linear(96 * 4, n_classes),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = x.view(x.size(0), -1)
        logits = self.out(x)

        return logits.squeeze()


@torch.no_grad()
def predict(dataloader, model):
    model.eval()
    predictions = np.array([])
    y_target = np.array([])
    for x_batch, y_batch in dataloader:
        y_target = np.hstack((y_target, y_batch.numpy().flatten()))
        x_batch = x_batch.to(device)
        lgts = model.forward(x_batch)
        probs = torch.sigmoid(lgts)
        preds = (probs > 0.5).type(torch.long).cpu()
        predictions = np.hstack((predictions, preds.numpy().flatten()))
        x_batch.cpu()
    model.train()
    return predictions.flatten(), y_target


def train(model, optimizer, loss_function, train_dataloader, val_dataloader, max_epochs=10, end_acc=0.95):
    losses = np.zeros(max_epochs)
    acc_val = []
    model.train()
    for epoch in range(max_epochs):
        with tqdm(total=len(train_dataloader), desc=f"Epoch {epoch}", unit="batch") as pbar:
            for it, (X_batch, y_batch) in enumerate(train_dataloader):
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                optimizer.zero_grad()
                outp = model.forward(X_batch)
                loss = loss_function(outp.squeeze(), y_batch)
                losses[epoch] += loss.item()
                loss.backward()
                optimizer.step()
                pbar.set_postfix(loss=loss.item())
                pbar.update()
                y_batch.cpu()
                X_batch.cpu()
                outp.cpu()

            losses[epoch] /= len(train_dataloader)
            Y_pred_val, Y_target_val = predict(val_dataloader, model)
            val_balanced_accuracy = balanced_accuracy_score(Y_target_val, Y_pred_val)
            print(f"Epoch {epoch}: average loss={losses[epoch]:.4f}, val balanced accuracy: {val_balanced_accuracy}")
            acc_val.append(val_balanced_accuracy)
            if val_balanced_accuracy >= end_acc:
                break
    return losses, acc_val


simpleNN = SimpleCnn(256, 1).to(device)
loss_function = nn.BCEWithLogitsLoss()
learning_rate = 0.001
optimizer = torch.optim.Adam(simpleNN.parameters(), lr=learning_rate)

max_epochs = 20
losses, acc_val = train(simpleNN, optimizer, loss_function, train_dataloader, test_dataloader, max_epochs, end_acc=0.95)


def plot_losses(losses):
    plt.figure(figsize=(12, 8))
    plt.plot(range(len(losses)), losses)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.show()


plot_losses(losses)

scheduler = StepLR(optimizer, step_size=3, gamma=0.1)
for epoch in range(max_epochs):
    train(simpleNN, optimizer, loss_function, train_dataloader, test_dataloader, max_epochs, end_acc=0.95)
    scheduler.step()


class OurOwnDataset(Dataset):
    def __init__(self, data, labels, transforms=None):
        self.data = data
        self.labels = labels
        self.transforms = transforms
        print(f'Found {len(self.data)} items')

    def __getitem__(self, i):
        image = self.data[i]
        if self.transforms:
            image = self.transforms(image)
        label = self.labels[i]
        return image, label

    def __len__(self):
        return len(self.data)


Y_pred_train, Y_target_train = predict(train_dataloader, simpleNN)
train_balanced_accuracy = balanced_accuracy_score(Y_target_train, Y_pred_train)
print(f"Train Balanced Accuracy: {train_balanced_accuracy}")

Y_pred_val, Y_target_val = predict(val_dataloader, simpleNN)
val_balanced_accuracy = balanced_accuracy_score(Y_target_val, Y_pred_val)
print(f"Val Balanced Accuracy: {val_balanced_accuracy}")

Y_test_pred, Y_test_target = predict(test_dataloader, simpleNN)
test_balanced_accuracy = balanced_accuracy_score(Y_test_target, Y_test_pred)
print(f"Test Balanced Accuracy: {test_balanced_accuracy}")
getitem_transforms = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip()
])
train_dataset_augmented = OurOwnDataset(X_train_t, Y_train_t, transforms=getitem_transforms)
plt.figure(figsize=(18, 6))
for i in range(6):
    j = 100 + 10 * i
    image, label = train_dataset_augmented.__getitem__(j)
    plt.subplot(2, 6, i + 1)
    plt.axis("off")
    imshow(image, title=f'{label}', plt_ax=plt)
plt.show()

train_dataloader_augmented = DataLoader(train_dataset_augmented, batch_size=128, sampler=sampler)
model2 = SimpleCnn(256, 1).to(device)
optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.005)
losses2, acc_val2 = train(model2, optimizer2, loss_function, train_dataloader_augmented, val_dataloader, max_epochs=50,
                          end_acc=0.95)
plot_losses(losses2)

Y_train_pred2, Y_train_target2 = predict(train_dataloader_augmented, model2)
balanced_accur = balanced_accuracy_score(Y_train_target2, Y_train_pred2)
print(f"Train Balanced Accuracy (Augmented): {balanced_accur}")

Y_pred_val2, Y_target_val2 = predict(val_dataloader, model2)
balanced_accur = balanced_accuracy_score(Y_target_val2, Y_pred_val2)
print(f"Val Balanced Accuracy (Augmented): {balanced_accur}")

model3 = models.efficientnet_v2_s(pretrained=True)
for par in model3.parameters():
    par.requires_grad = False

num_features = model3.classifier[1].in_features
model3.classifier[1] = nn.Sequential(
    nn.Flatten(),
    nn.Linear(num_features, 96 * 2),
    nn.ReLU(),
    nn.BatchNorm1d(96 * 2),
    nn.Dropout(p=0.2),
    nn.Linear(96 * 2, 96),
    nn.ReLU(),
    nn.BatchNorm1d(96),
    nn.Dropout(p=0.1),
    nn.Linear(96, 1),
)

model3 = model3.to(device)
optimizer3 = torch.optim.Adam(model3.classifier[1].parameters(), lr=0.05)

losses3, acc_val3 = train(model3, optimizer3, loss_function, train_dataloader, val_dataloader, max_epochs=90,
                          end_acc=0.90)
plot_losses(losses3)

Y_pred_train3, Y_train_target3 = predict(train_dataloader, model3)
balanced_accuracy = balanced_accuracy_score(Y_train_target3, Y_pred_train3)
print(f"Train Balanced Accuracy (EfficientNet): {balanced_accuracy}")

Y_pred_val3, Y_target_val3 = predict(val_dataloader, model3)
balanced_accuracy = balanced_accuracy_score(Y_target_val3, Y_pred_val3)
print(f"Val Balanced Accuracy (EfficientNet): {balanced_accuracy}")


def predict(dataloader, model):
    model.eval()
    predictions = np.array([])
    y_target = np.array([])
    for x_batch, y_batch in dataloader:
        y_target = np.hstack((y_target, y_batch.numpy().flatten()))
        x_batch = x_batch.to(device)
        lgts = model(x_batch)
        probs = torch.sigmoid(lgts)
        preds = (probs > 0.5).type(torch.long).cpu()
        predictions = np.hstack((predictions, preds.numpy().flatten()))
        x_batch.cpu()
    model.train()
    return predictions.flatten(), y_target


Y_pred_train, Y_target_train = predict(train_dataloader, simpleNN)

train_balanced_accuracy = balanced_accuracy_score(Y_target_train, Y_pred_train)
print(f"Train Balanced Accuracy: {train_balanced_accuracy}")

Y_pred_val, Y_target_val = predict(val_dataloader, simpleNN)

val_balanced_accuracy = balanced_accuracy_score(Y_target_val, Y_pred_val)
print(f"Val Balanced Accuracy: {val_balanced_accuracy}")

Y_test_pred, Y_test_target = predict(test_dataloader, simpleNN)
test_balanced_accuracy = balanced_accuracy_score(Y_test_target, Y_test_pred)
print(f"Test Balanced Accuracy: {test_balanced_accuracy}")


class OurOwnDataset(Dataset):
    def __init__(self, data, labels, transforms=None):
        self.data = data
        self.labels = labels
        self.transforms = transforms

        print(f'Found {len(self.data)} items')

    def __getitem__(self, i):
        image = self.data[i]

        if self.transforms:
            image = self.transforms(image)

        label = self.labels[i]

        return image, label

    def __len__(self):
        return len(self.df)


getitem_transforms = Compose([
    RandomRotation(15),
    RandomHorizontalFlip()
])

train_dataset_augmented = OurOwnDataset(X_train_t, Y_train_t, transforms=getitem_transforms)

plt.figure(figsize=(18, 6))
for i in range(6):
    j = 100 + 10 * i;
    image, label = train_dataset_augmented.__getitem__(j)

    plt.subplot(2, 6, i + 1)
    plt.axis("off")

    imshow(image, title=f'{label}', plt_ax=plt)
plt.show();

train_dataloader_augmented = DataLoader(train_dataset_augmented, batch_size=128, sampler=sampler)

model2 = SimpleCnn(256, 1).to(device)
optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.005)
losses2, acc_val2 = train(model2, optimizer2, loss_function, train_dataloader_augmented, val_dataloader, max_epochs=40,
                          end_acc=0.95)
plot_losses(losses2)

Y_train_pred2, Y_train_target2 = predict(train_dataloader_augmented, model2)
balanced_accur = balanced_accuracy_score(Y_train_target2, Y_train_pred2)
print(f"Train Balanced Accuracy (Augmented): {balanced_accur}")

Y_pred_val2, Y_target_val2 = predict(val_dataloader, model2)
balanced_accur = balanced_accuracy_score(Y_target_val2, Y_pred_val2)
print(f"Val Balanced Accuracy (Augmented): {balanced_accur}")

model3 = models.efficientnet_v2_s(pretrained=True)
for par in model3.parameters():
    par.requires_grad = False

num_features = model3.classifier[1].in_features
model3.classifier[1] = nn.Sequential(nn.Flatten(),
                                     nn.Linear(num_features, 96 * 2),
                                     nn.ReLU(),
                                     nn.BatchNorm1d(96 * 2),
                                     nn.Dropout(p=0.2),
                                     nn.Linear(96 * 2, 96),
                                     nn.ReLU(),
                                     nn.BatchNorm1d(96),
                                     nn.Dropout(p=0.1),
                                     nn.Linear(96, 1), )

model3 = model3.to(device)
optimizer3 = torch.optim.Adam(model3.classifier[1].parameters(), lr=0.005)

losses3, acc_val3 = train(model3, optimizer3, loss_function, train_dataloader, val_dataloader, max_epochs=90,
                          end_acc=0.9)
plot_losses(losses3)

Y_pred_train3, Y_train_target3 = predict(train_dataloader, model3)
balanced_accuracy = balanced_accuracy_score(Y_train_target3, Y_pred_train3)
print(f"Train Balanced Accuracy (EfficientNet): {balanced_accuracy}")

Y_pred_val3, Y_target_val3 = predict(val_dataloader, model3)
balanced_accuracy = balanced_accuracy_score(Y_target_val3, Y_pred_val3)
print(f"Val Balanced Accuracy (EfficientNet): {balanced_accuracy}")
