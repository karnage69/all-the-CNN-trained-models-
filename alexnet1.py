import torch 
import torch.nn as nn
from torchvision import datasets,transforms
import torch.optim as optim
from torch.utils.data import DataLoader
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])
train= datasets.CIFAR100(root="./data",download=True,train=True,transform=transform)
test= datasets.CIFAR100(root="./data",download=True,train=False,transform=transform)
train_data= DataLoader(train,shuffle=True,batch_size=64)
test__data= DataLoader(test,shuffle=True,batch_size=64)
print("cuda", torch.cuda.is_available())
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
class AlexNet(nn.Module):
    def __init__(self,num_class=100):
        super(AlexNet,self).__init__()
        self.features=nn.Sequential(
            nn.Conv2d(3,64,kernel_size=3,padding=1),#64x64ho gya
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),#32x32
            
            nn.Conv2d(64,128,kernel_size=3,padding=1),#32x32
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),#16x16
            
            nn.Conv2d(128,256,kernel_size=3,padding=1),#16X16
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),#8x8

            nn.Conv2d(256,512,kernel_size=3,padding=1),#8x8
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool2d(1),#1x1
            
            )
        self.classifier = nn.Linear(512, num_class)

        
    
    
    
    def forward(self,x):
        x= self.features(x)
        x= torch.flatten(x,1)
        x= self.classifier(x)
        return x
    
model = AlexNet(num_class=100).to(device)
criterion= nn.CrossEntropyLoss()
optimizer= optim.Adam(model.parameters(),lr=0.001)
for epochs in range(20):
    model.train()
    correct,total=0,0
    for images, labels in train_data:
        images, labels = images.to(device),labels.to(device)
        optimizer.zero_grad()
        output= model(images)
        loss= criterion(output,labels)
        loss.backward()
        optimizer.step()
        _,predict = torch.max(output,1)
        total += labels.size(0)
        correct+=(predict == labels).sum().item()
print(f"Epoch {epochs+1}: Train Accuracy = {correct/total:.4f}")
torch.save(model.state_dict(), "alexnet1.pth")
print("saved")