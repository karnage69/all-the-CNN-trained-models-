import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
transform= transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])
train_data= datasets.CIFAR100(root="./data",download=True,train=True,transform=transform)
test_data= datasets.CIFAR100(root="./data",download=True,train=False,transform=transform)
train_loader= DataLoader(train_data,shuffle=True,batch_size=64)
test_loader= DataLoader(test_data,shuffle=False,batch_size=64)
print("cuda",torch.cuda.is_available())
device= torch.device("cuda" if torch.cuda.is_available() else"cpu")
class VGG11(nn.Module):
    def __init__(self,num_classes=100):
        super(VGG11,self).__init__()
        self.features= nn.Sequential(
            nn.Conv2d(3,64,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64,128,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            
            
            nn.Conv2d(128,256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            
            nn.Conv2d(256,512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            
            nn.Conv2d(512,512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),

            nn.AdaptiveAvgPool2d(1)
)
        self.classifier=nn.Linear(512,num_classes)
    def forward(self,x):
        x= self.features(x)
        x = torch.flatten(x, 1)
        x= self.classifier(x)
        return x
model = VGG11(num_classes=100).to(device)
criterion = nn.CrossEntropyLoss()
optimizer= optim.Adam(model.parameters(),lr=0.001)
for epochs in range(20):
    model.train()
    train_loss,correct,total=0.0,0,0
    for images, labels in train_loader:
        images,labels = images.to(device),labels.to(device)
        optimizer.zero_grad()
        output= model(images)
        loss = criterion(output,labels)
        loss.backward()
        optimizer.step()
        train_loss+= loss.item()
        _, predict =torch.max(output.data,1)
        total+= labels.size(0)
        correct+= (predict==labels).sum().item()
print("accuracy", correct/total)
torch.save(model.state_dict(),"VGG11_light_weights.pth")  
print("saved")
