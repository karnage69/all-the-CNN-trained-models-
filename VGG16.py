import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
transform= transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(    (0.5071, 0.4867, 0.4408),
    (0.2675, 0.2565, 0.2761))
])
train_data= datasets.CIFAR100(root="./data", train=True,download=True,transform=transform)
test_data= datasets.CIFAR100(root="./data", train=False,download=True,transform=transform)
train_loader= DataLoader(train_data,shuffle=True,batch_size=64)
test_loader= DataLoader(test_data,shuffle=False,batch_size=64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class VGG16(nn.Module):
    def __init__(self,num_class=100):
        super(VGG16,self).__init__()
        self.functional= nn.Sequential(
            nn.Conv2d(3,64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            
            nn.Conv2d(64,128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            
            nn.Conv2d(128,256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            
            nn.Conv2d(256,512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            
            nn.Conv2d(512,512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),

            nn.AdaptiveAvgPool2d(1)
)
        self.classifier= nn.Linear(512,num_class)
    def forward(self,x):
        x= self.functional(x)
        x= torch.flatten(x,1)
        x=self.classifier(x)
        return x
model= VGG16(num_class=100).to(device)
criterion= nn.CrossEntropyLoss()
optimizer= optim.SGD(model.parameters(),lr=0.01,momentum=0.9,weight_decay=5e-4)
for epochs in range(20):
    model.train()
    correct,total,total_loss= 0,0,0.0
    for images,labels in train_loader:
        images,labels= images.to(device),labels.to(device)
        optimizer.zero_grad()
        output= model(images)
        loss= criterion(output,labels)
        loss.backward()
        optimizer.step()
        total_loss+= loss.item()
        _, predict =torch.max(output,1)
        total+= labels.size(0)
        correct+= (predict==labels).sum().item()
    acc = (output.argmax(1) == labels).float().mean().item()
    print(f"Epoch {epochs+1} | Loss: {total_loss/len(train_loader):.4f} | Train Acc: {correct/total:.4f}")
torch.save(model.state_dict(),"VGG16_light_weights.pth")  
print("saved")