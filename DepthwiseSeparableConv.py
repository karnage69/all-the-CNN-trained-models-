import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform= transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])
train_data= datasets.CIFAR100(root="./data", download=True,transform=transform,train=True)
test_data= datasets.CIFAR100(root="./data", download=True,transform=transform,train=False)
train_loader= DataLoader(train_data,shuffle=True,batch_size=64)
test_loader= DataLoader(test_data,shuffle=False,batch_size=64)
class Depthwiseseparableconv(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1):
        super().__init__()
        self.depthwise=nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            bias=False
        )
        self.pointwise=nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    def forward(self,x):
        x=self.depthwise(x)
        x=self.pointwise(x)
        x=self.bn(x)
        x=self.relu(x)
        return x
class CNN_DS(nn.Module):
    def __init__(self):
        
        super().__init__()
        self.features= nn.Sequential(
        Depthwiseseparableconv(3,64),
        nn.MaxPool2d(2,2),
        Depthwiseseparableconv(64,128),
        nn.MaxPool2d(2,2),
        Depthwiseseparableconv(128,256),
        nn.MaxPool2d(2,2)
        
)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256*4*4,512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 100)  
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model=CNN_DS().to(device)
criterion=nn.CrossEntropyLoss()
optimizer= optim.Adam(model.parameters(),lr=0.001)
for epoch in range(20):
    model.train()
    train_loss,correct,total=0.0,0,0
    for image,labels in train_loader:
        image,labels= image.to(device),labels.to(device)
        optimizer.zero_grad()
        output= model(image)
        loss= criterion(output,labels)
        loss.backward()
        optimizer.step()
        train_loss+= loss.item()
        _,predict= torch.max(output,1)
        total+= labels.size(0)
        correct+= (predict== labels).sum().item()
    print("accuracy",correct/total)
torch.save(model.state_dict(), "dwsc_weights.pth")
print("Model saved")