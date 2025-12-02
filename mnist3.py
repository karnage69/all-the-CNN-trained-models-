#just changing the filters and padding from previous ones
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
transfrom= transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
])
train_data= datasets.FashionMNIST(root="./data",train=True,transform=transfrom,download=True)
test_data= datasets.FashionMNIST(root="./data",train=True,transform=transfrom,download=True)
train_loader=DataLoader(train_data,batch_size=64,shuffle=True)
test_loader=DataLoader(test_data,batch_size=64,shuffle=False)
print("cuda",torch.cuda.is_available())
device=torch.device("cuda"if torch.cuda.is_available() else "cpu")
print(f"{device}")
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1= nn.Conv2d(1,32,kernel_size=5,padding=2)
        self.conv2= nn.Conv2d(32,64,kernel_size=5,padding=2)
        self.conv3= nn.Conv2d(64,128,kernel_size=5,padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128*7*7,128)
        self.fc2 = nn.Linear(128,10)
    def forward(self,x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))   
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0),-1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
model = CNN().to(device)
criterion= nn.CrossEntropyLoss()
optimzer= optim.Adam(model.parameters(),lr=0.001)
for epochs in range(20):
    model.train()
    train_loss,correct,total=-0.0,0,0
    for images, labels in train_loader:
        images,labels=images.to(device),labels.to(device)
        optimzer.zero_grad()
        output= model(images)
        loss =criterion(output,labels)
        loss.backward()
        optimzer.step()
        train_loss +=loss.item()
        _,predicted=torch.max(output.data,1)
        total += labels.size(0)
        correct +=( predicted==labels).sum().item()
print("Test Accuracy:", correct/total)
torch.save(model.state_dict(), "mnist3_weights.pth")
print("Model saved in this folder.")
