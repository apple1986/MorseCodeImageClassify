import torch
import torch.nn.functional as F  # Parameterless functions, like (some) activation functions
import torchvision.datasets as datasets  # Standard datasets
import torchvision.transforms as transforms  # Transformations we can perform on our dataset for augmentation
from torch import optim  # For optimizers like SGD, Adam, etc.
from torch import nn  # All neural network modules
from torch.utils.data import Dataset, DataLoader # Gives easier dataset managment by creating mini batches etc.
from tqdm import tqdm  # For nice progress bar!
import os
import numpy as np
import random
import cv2

# define seed
def seed_everything(seed=2024):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Simple CNN
class SimpleCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=26):
        super(SimpleCNN, self).__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=8,
                    kernel_size=3,
                    stride=1,
                    padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(
                    in_channels=8,
                    out_channels=16,
                    kernel_size=3,
                    stride=1,
                    padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )  
        self.stage3 = nn.Sequential(
            nn.Conv2d(
                    in_channels=16,
                    out_channels=32,
                    kernel_size=3,
                    stride=1,
                    padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ) 
        self.stage4 = nn.Sequential(
            nn.Conv2d(
                    in_channels=32,
                    out_channels=64,
                    kernel_size=3,
                    stride=1,
                    padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ) 
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(14 * 14 * 64, 1000),
            nn.ReLU(),
            nn.Linear(1000, num_classes),
        )


    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.classifier(x)
        return x



# Hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Set device
seed_everything(seed=2024) # set random seed
in_channels = 3
num_classes = 26
learning_rate = 3e-4 # karpathy's constant
batch_size = 8
num_epochs = 3


# Build a dataset class
class ImageFolder(Dataset):
    def __init__(self, root_dir, transform_type="train"):
        super(ImageFolder, self).__init__()
        self.data = []
        self.root_dir = root_dir
        self.transform_type = transform_type
        self.class_names = os.listdir(root_dir)

        # Load Data
        self.transform_train = transforms.Compose(
            [  # Compose makes it possible to have many transforms
                transforms.ToTensor(),  # Finally converts PIL image to tensor so we can train w. pytorch
                transforms.Resize((224, 224)),  # Resizes (32,32) to (36,36)
                transforms.ColorJitter(brightness=0.5),  # Change brightness of image
                transforms.RandomRotation(
                    degrees=10
                ),  # Perhaps a random rotation from -45 to 45 degrees
                transforms.RandomGrayscale(p=0.2),  # Converts to grayscale with probability 0.2
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
                ),  # Note: change this value
            ]
        )
        self.transform_test = transforms.Compose(
            [  # Compose makes it possible to have many transforms
                transforms.ToTensor(),  # Finally converts PIL image to tensor so we can train w. pytorch
                transforms.Resize((224, 224)),  # Resizes (32,32) to (36,36)
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
                ),  # Note: change this value
            ]
        )


        for index, name in enumerate(self.class_names):
            files = os.listdir(os.path.join(root_dir, name))
            self.data += list(zip(files, [index] * len(files)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_file, label = self.data[index]
        root_and_dir = os.path.join(self.root_dir, self.class_names[label])
        image = cv2.imread(os.path.join(root_and_dir, img_file), cv2.IMREAD_COLOR)
        
        if self.transform_type is "train":
            image = self.transform_train(image)

        if self.transform_type is "test":
            image = self.transform_train(image)

        return image, label


train_dataset = ImageFolder(root_dir="MorseCode/MorseCode_Normal_Train_Test/MorseCode_train", transform_type="train")
test_dataset = ImageFolder(root_dir="MorseCode/MorseCode_Normal_Train_Test/MorseCode_train", transform_type="test")
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Initialize network
model = SimpleCNN(in_channels=in_channels, num_classes=num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train Network
for epoch in range(num_epochs):
    loss_total = []
    for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)
        # forward
        scores = model(data)
        loss = criterion(scores, targets)

        # backward
        optimizer.zero_grad()
        loss.backward()
        # gradient descent or adam step
        optimizer.step()

# Check accuracy on training & test to see how good our model
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    model.train()
    return num_correct / num_samples


print(f"Accuracy on training set: {check_accuracy(train_loader, model)*100:.2f}")
print(f"Accuracy on test set: {check_accuracy(test_loader, model)*100:.2f}")