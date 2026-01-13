import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim 
device= torch.device("cuda" if torch.cuda.is_available() else"cpu")
transform= transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])
train_data= datasets.CIFAR100(root="./data",download=True,train=True,transform=transform)
test_data= datasets.CIFAR100(root="./data",download=True,train=False,transform=transform)
train_loader= DataLoader(train_data,shuffle=True,batch_size=64)
test_loader= DataLoader(test_data,shuffle=False,batch_size=64)
class Bottleneck(nn.Module):
    expansion=4
    def __init__(self, in_channel, out_channel, stride=1, identity_downsample=None):
        super().__init__()
        self.conv1= nn.Conv2d(in_channel,out_channel,kernel_size=1,stride=1,padding=0,bias=False)
        self.bn1= nn.BatchNorm2d(out_channel)
        self.conv2= nn.Conv2d(out_channel,out_channel,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn2= nn.BatchNorm2d(out_channel)
        self.conv3= nn.Conv2d(out_channel,out_channel*self.expansion,kernel_size=1,stride=1,padding=0,bias=False)
        self.bn3= nn.BatchNorm2d(out_channel*self.expansion)
        self.relu= nn.ReLU(inplace=True)
        self.identity_downsample= identity_downsample
        
        
    def forward(self,x):
        identity= x
        x= self.conv1(x)
        x= self.bn1(x)
        x= self.relu(x)
        x= self.conv2(x)
        x= self.bn2(x)
        x= self.relu(x)
        x= self.conv3(x)
        x= self.bn3(x)
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x
class ResNet50(nn.Module):
    def __init__(self,block,layers,num_classes=100):
        super().__init__()
        self.in_channels=64
        self.conv1= nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn1= nn.BatchNorm2d(64)
        self.relu= nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1= self._make_layer(block,64,layers[0],stride=1)
        self.layer2= self._make_layer(block,128,layers[1],stride=2)
        self.layer3= self._make_layer(block,256,layers[2],stride=2)
        self.layer4= self._make_layer(block,512,layers[3],stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
    def _make_layer(self, block,out_channel,num_blocks,stride):
        identity_downsample=None
        layer=[]
        if stride !=1 or self.in_channels != out_channel*block.expansion:
            identity_downsample=nn.Sequential(
                nn.Conv2d(self.in_channels,out_channel*block.expansion,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(out_channel*block.expansion)
            )
        layer.append(
                block(self.in_channels,out_channel,stride,identity_downsample=identity_downsample)
            )
        self.in_channels = out_channel*block.expansion
        for _ in range(1, num_blocks):
            layer.append(
                block(self.in_channels, out_channel, stride=1)
                )
        return nn.Sequential(*layer)
    def forward(self, x):
     x = self.relu(self.bn1(self.conv1(x)))
     x = self.maxpool(x)

     x = self.layer1(x)
     x = self.layer2(x)
     x = self.layer3(x)
     x = self.layer4(x)

     x = self.avgpool(x)
     x = torch.flatten(x, 1)
     x = self.fc(x)

     return x

model= ResNet50(block=Bottleneck,layers=[3,4,6,3],num_classes=100).to(device)
criterion= nn.CrossEntropyLoss()
Optimizer= optim.Adam(model.parameters(),lr=0.001)
for epochs in range(20):
    model.train()
    train_loss,correct,total=0.0,0,0
    for images, labels in train_loader:
        images,labels = images.to(device),labels.to(device)
        Optimizer.zero_grad()
        output= model(images)
        loss= criterion(output,labels)
        loss.backward()
        Optimizer.step()
        train_loss+= loss.item()
        __, predict = torch.max(output,1)
        total+=labels.size(0)
        correct+= (predict==labels).sum().item()
        print("accuracy", correct/total)
torch.save(model.state_dict(), "resnet18.pth")
print("saved")
