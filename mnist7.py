#using Gloabal average pooling to flattening
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
])
train_data= datasets.FashionMNIST(root="./data",train=True,download=True,transform=transform)
test_data= datasets.FashionMNIST(root="./data",train=False,download=True,transform=transform)
train_loader=DataLoader(train_data,shuffle=True,batch_size=64)
test_loader=DataLoader(test_data,shuffle=False,batch_size=64)
print("cuda", torch.cuda.is_available())
device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
class CNN_GAP(nn.Module):
    def __init__(self):
        super(CNN_GAP,self).__init__()
        self.conv1= nn.Conv2d(1,32,kernel_size=3,stride=2,padding=1)
        self.conv2= nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1)
        self.conv3= nn.Conv2d(64,128,kernel_size=3,stride=2,padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(128, 10)#only last layer need channel wrna humko fc ki zarurati nahi pdti
        self.dropout = nn.Dropout(0.3)
    def forward(self,x):
        x= torch.relu(self.conv1(x))#[28x28 -> 14x14]
        x= torch.relu(self.conv2(x))#[14x14 -> 14x14  (as stride in this is 1)]
        x= self.pool(torch.relu(self.conv3(x)))#[14x14->7x7 ,7x7-> 3x3]
        x= self.gap(x) #output = [bacthsize,128,1,1]
        x = x.view(x.size(0), -1)#output=[batchsize,128(removed 1,1)]
        x = self.dropout(x)
        x = self.fc(x)
        return x
model= CNN_GAP().to(device)
criterion= nn.CrossEntropyLoss()
optimizer= optim.Adam(model.parameters(),lr=0.001)
for epochs in range(20):
    model.train()
    train_loss,correct,total=0.0,0,0
    for images,labels in train_loader:
        images,labels=images.to(device),labels.to(device)
        optimizer.zero_grad()
        output= model(images)
        loss=criterion(output,labels)
        loss.backward()
        optimizer.step()
        train_loss+= loss.item()
        _, predict= torch.max(output.data,1)
        total+= labels.size(0)
        correct +=( predict==labels).sum().item()
print("accuracy",correct/total)
torch.save(model.state_dict(), "mnist7_weights.pth")
print("Model saved in this folder.")