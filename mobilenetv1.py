import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
transform= transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])
print("cuda", torch.cuda.is_available())
device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_data= datasets.CIFAR100(root="./data",transform=transform,download=True,train=True)
test_data= datasets.CIFAR100(root="./data",transform=transform,download=True,train=False)
train_loader= DataLoader(train_data,shuffle=True,batch_size=64)
test_loader= DataLoader(test_data,shuffle=False,batch_size=64)
class DepthWiseSeprableconv(nn.Module):
    def __init__(self,in_channels,out_channels,stride):
        super().__init__()
        self.depthwise= nn.Sequential(nn.Conv2d(in_channels,in_channels,kernel_size=3,stride=stride,padding=1,groups=in_channels,bias=False),
                                      nn.BatchNorm2d(in_channels),nn.ReLU(inplace=True))
        self.pointwise= nn.Sequential(nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1,padding=0,bias=False),
                                      nn.BatchNorm2d(out_channels),nn.ReLU(inplace=True))
    def forward(self,x):
        x= self.depthwise(x)
        x= self.pointwise(x)
        return x 
class MobileNetV1(nn.Module):
    def __init__(self,num_classes=1000):
        nn.Module.__init__(self)
        self.features= nn.Sequential(
            nn.Conv2d(3,32,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            DepthWiseSeprableconv(32,64,stride=1),
            DepthWiseSeprableconv(64,128,stride=2),
            DepthWiseSeprableconv(128,128,stride=1),
            DepthWiseSeprableconv(128,256,stride=2),
            DepthWiseSeprableconv(256,256,stride=1),
            DepthWiseSeprableconv(256,512,stride=2),

            DepthWiseSeprableconv(512,512,stride=1),
            DepthWiseSeprableconv(512,512,stride=1),
            DepthWiseSeprableconv(512,512,stride=1),
            DepthWiseSeprableconv(512,512,stride=1),
            DepthWiseSeprableconv(512,512,stride=1),

            DepthWiseSeprableconv(512,1024,stride=2),
            DepthWiseSeprableconv(1024,1024,stride=1),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(1024, num_classes)
    def forward(self,x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


model = MobileNetV1(num_classes=100).to(device)
critarion= nn.CrossEntropyLoss()
optimizer= optim.Adam(model.parameters(),lr=0.001)
for epoch in range(20):
    model.train()
    total,correct,train_loss=0,0,0.0
    for images, labels in train_loader:
        images,labels= images.to(device),labels.to(device)
        optimizer.zero_grad()
        output=model(images)
        loss=critarion(output,labels)
        loss.backward()
        optimizer.step()
        train_loss+= loss.item()
        _,predict= torch.max(output.data,1)
        total+= labels.size(0)
        correct+=(predict==labels).sum().item()
print(f"Epoch {epoch+1} accuracy: {correct/total}")
torch.save(model.state_dict(),"MobileNetv1.pth")
print("model has been saved")
