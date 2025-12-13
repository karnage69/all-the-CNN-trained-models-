import torch
import torch.nn as nn
import torch.optim as optim
from  torchvision import datasets, transforms
from torch.utils.data import DataLoader
transform= transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])
train_data= datasets.CIFAR100(root="./data",train=True,transform=transform,download=True)
test_data= datasets.CIFAR100(root="./data",train=False,transform=transform,download=True)
train_loader= DataLoader(train_data,shuffle=True,batch_size=64)
test_loader= DataLoader(test_data,shuffle=False,batch_size=64)
print("cuda", torch.cuda.is_available())
device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{device}")
class LeNet(nn.Module):
    def __init__(self, num_classes=100):
        super(LeNet,self).__init__()
        self.conv1 = nn.Conv2d(3,6, kernel_size=5) 
        self.pool = nn.AvgPool2d(2,2)
        self.conv2 = nn.Conv2d(6,12, kernel_size=5) 
        self.fc1 = nn.Linear(12*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,num_classes)
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))   
        x = self.pool(torch.relu(self.conv2(x)))   
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
model= LeNet().to(device)
criterion= nn.CrossEntropyLoss()
optimizer= optim.Adam(model.parameters(),lr=0.001)
for epochs in range(20):
    model.train()
    train_loss,correct,total=0.0,0,0
    for images, labels in train_loader:
        images,labels= images.to(device),labels.to(device)
        optimizer.zero_grad()
        output= model(images)
        loss= criterion(output,labels)
        loss.backward()
        optimizer.step()
        train_loss+= loss.item()
        _,predicted=torch.max(output,1)
        total+=labels.size(0)
        correct+= (predicted==labels).sum().item()
print(f"test accuracy{100 * correct/total:.2f}")
torch.save(model.state_dict(), "Lenet1_weights.pth")
print("model saved")