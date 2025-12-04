#just adding random dropout
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
transform= transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
])
train_data= datasets.FashionMNIST(root="./data",download=True,train=True,transform=transform)
test_data= datasets.FashionMNIST(root="./data",download=True,train=False,transform=transform)
train_loader=DataLoader(train_data,shuffle=True,batch_size=64)
test_loader=DataLoader(test_data,shuffle=False,batch_size=64)
print("cuda",torch.cuda.is_available())
device=torch.device("cuda"if torch.cuda.is_available() else "cpu")
print(f"{device}")
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1= nn.Conv2d(1,32,kernel_size=5,padding=2)
        self.bn1= nn.BatchNorm2d(32)
        self.conv2= nn.Conv2d(32,64,kernel_size=5,padding=2)
        self.bn2= nn.BatchNorm2d(64)
        self.conv3= nn.Conv2d(64,128,kernel_size=5,padding=2)
        self.bn3= nn.BatchNorm2d(128)
        self.pool=nn.MaxPool2d(2,2)
        self.dropout1= nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128*7*7,128)
        self.bn4 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128,10)
    def forward(self,x):
        x= self.pool(torch.relu(self.bn1(self.conv1(x))))
        x= self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.dropout1(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.bn4(self.fc1(x)))
        x = self.dropout2(x)
        x = self.fc2(x)  
        return x
model= CNN().to(device)
criterion= nn.CrossEntropyLoss()
optimizer= optim.Adam(model.parameters(),lr=0.001)
for epochs in range(20):
    model.train()
    train_loss,correct,total=0.0,0,0
    for images,label in train_loader:
        images,label= images.to(device),label.to(device)
        optimizer.zero_grad()
        output=model(images)
        loss=criterion(output,label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _,predicted=torch.max(output.data,1)
        total += label.size(0)
        correct +=( predicted==label).sum().item()
print("accuracy",correct/total)
torch.save(model.state_dict(), "mnist5_weights.pth")
print("Model saved in this folder.")