import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
transform= transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])
device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_data= datasets.CIFAR100(root="./data",download=True,train=True,transform=transform)
test_data= datasets.CIFAR100(root="./data",download=True,train=False,transform=transform)
train_loader= DataLoader(train_data,shuffle=True,batch_size=64)
test_loader= DataLoader(test_data,shuffle=False,batch_size=64)
class MLPBlock(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.block= nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels,kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels,kernel_size=1),
            nn.ReLU(inplace=True),

        )
    def forward(self,x):
        return self.block(x)
class NiN(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.features = nn.Sequential(
            MLPBlock(3,192),
            nn.MaxPool2d(2,2),
            MLPBlock(192,256),
            nn.MaxPool2d(2,2)
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
    def forward(self,x):
        x=self.features(x)
        x=self.gap(x)
        x = x.view(x.size(0), -1)
        return x
model= NiN(num_classes=100).to(device)
citerion= nn.CrossEntropyLoss()
optimizer= optim.Adam(model.parameters(),lr=0.001)
for epochs in range(20):
    model.train()
    total,correct,train_loss=0,0,0.0
    for images, labels in train_loader:
        images,labels= images.to(device),labels.to(device)
        optimizer.zero_grad()
        output= model(images)
        loss= citerion(output,labels)
        loss.backward()
        optimizer.step()
        train_loss+= loss.item()
        _, predict = torch.max(output,1)
        total+= labels.size(0)
        correct+= (predict==labels).sum().item()
        print("accuracy", correct/total)
torch.save(model.state_dict(),"NiNweights.pth")  
print("saved")