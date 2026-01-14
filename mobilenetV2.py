import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
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
class InvertedResidual(nn.Module):
    def __init__(self,in_ch,out_ch,expansion,stride):
        super().__init__()
        new_in= in_ch*expansion
        self.residual =(stride == 1 and in_ch== out_ch)
        layers=[]
        #expanding
        if expansion != 1:
            layers.append(nn.Conv2d(in_ch,new_in,kernel_size=1,bias=False))
            layers.append(nn.BatchNorm2d(new_in))
            layers.append(nn.ReLU(inplace=True))
        #processing
        layers.append(nn.Conv2d(new_in,new_in,kernel_size=3,stride=stride,padding=1,groups=new_in,bias=False))
        layers.append(nn.BatchNorm2d(new_in))
        layers.append(nn.ReLU(inplace=True))
        #compressing
        layers.append(nn.Conv2d(new_in,out_ch,kernel_size=1,bias=False))
        layers.append(nn.BatchNorm2d(out_ch))
        #doing it in seq and making it a block


        self.block = nn.Sequential(*layers)
    def forward(self,x):
        if self.residual:
            return x + self.block(x)
        else:
            return self.block(x)
class MobileNetV2(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()

        self.features = nn.Sequential(
            
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),

            
            InvertedResidual(32,  16, stride=1, expansion=1),

            InvertedResidual(16,  24, stride=2, expansion=6),
            InvertedResidual(24,  24, stride=1, expansion=6),

            InvertedResidual(24,  32, stride=2, expansion=6),
            InvertedResidual(32,  32, stride=1, expansion=6),
            InvertedResidual(32,  32, stride=1, expansion=6),

            InvertedResidual(32,  64, stride=2, expansion=6),
            InvertedResidual(64,  64, stride=1, expansion=6),
            InvertedResidual(64,  64, stride=1, expansion=6),
            InvertedResidual(64,  64, stride=1, expansion=6),

            InvertedResidual(64,  96, stride=1, expansion=6),
            InvertedResidual(96,  96, stride=1, expansion=6),
            InvertedResidual(96,  96, stride=1, expansion=6),

            InvertedResidual(96, 160, stride=2, expansion=6),
            InvertedResidual(160,160, stride=1, expansion=6),
            InvertedResidual(160,160, stride=1, expansion=6),

            InvertedResidual(160,320, stride=1, expansion=6),
            nn.Conv2d(320, 1280, kernel_size=1, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

model = MobileNetV2(num_classes=100).to(device)
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
torch.save(model.state_dict(),"MobileNetv2.pth")
print("model has been saved")