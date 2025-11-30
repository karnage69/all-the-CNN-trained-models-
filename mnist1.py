import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
print("CUDA available :", torch.cuda.is_available())
devices= torch.device("cuda"if torch.cuda.is_available() else"cpu")
print(f"{devices} using")
transform= transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_data = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_data = datasets.MNIST(root="./data", train= False,download=True, transform=transform)

train_loader =DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data,batch_size=64,shuffle=False)
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self). __init__()
        self.conv1= nn.Conv2d(1,32,kernel_size=3,padding=1)
        self.conv2= nn.Conv2d(32,32,kernel_size=3,padding=1)
        self.pool =nn.MaxPool2d(2)
        self.dropout=nn.Dropout(0.5)
        self.fc = nn.Linear(32 * 7 * 7, 10)
    def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = self.pool(x)
        
            x = torch.relu(self.conv2(x))
            x = self.pool(x)
        
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x  
model = CNN().to(devices)     
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
#training loop 
best_loss= float("inf")
for epoch in range(20):
    model.train()
    train_loss, correct,total = 0.0,0,0
    for images,labels in train_loader:
        images,labels= images.to(devices),labels.to(devices)
        optimizer.zero_grad()
        outputs= model(images)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        #tracking
        train_loss += loss.item()
        _,predicted = torch.max(outputs.data,1)
        total += labels.size(0)
        correct +=(predicted == labels).sum().item()
print(f"Test Accuracy: {100 * correct/total:.2f}%")
torch.save(model.state_dict(), "mnist1_weights.pth")
print("Model saved in this folder.")
