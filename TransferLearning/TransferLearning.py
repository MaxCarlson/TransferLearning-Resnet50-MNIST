import torch
from torch import nn
from torchvision import datasets, models
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision

if torch.cuda.is_available():  
    dev = "cuda:0" 
else:  
    dev = "cpu"  
device = torch.device('cpu')
#device = torch.device(dev) 

transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Resize((224,224))])
     #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batchSize = 128

trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

indicies = []
imagesPerCatagory = 10
catagoriesCount = {x : 0 for x in range(10)}
for i in range(len(trainset)):
    if catagoriesCount[trainset[i][1]] < imagesPerCatagory:
        catagoriesCount[trainset[i][1]] += 1
        indicies.append(i)
    
    filledCatagories = 0
    for k, v in catagoriesCount.items():
        if v == imagesPerCatagory:
            filledCatagories += 1
        else:
            break
    if filledCatagories == 10:
        break

subset = torch.utils.data.Subset(trainset, indicies)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchSize, shuffle=False, num_workers=2)

testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batchSize, shuffle=False, num_workers=2)

model = models.resnet50(pretrained=True)
print(model)

model.fc = nn.Linear(in_features=2048, out_features=10, bias=True)
print(model)

# Freeze all weights but the fully connected layer
for i, (name, param) in enumerate(model.named_parameters()):
    if name != 'fc.weight' and name != 'fc.bias':
        param.requires_grad = False

epochs  = 2
lr      = 0.01
#loss_fn = nn.MSELoss()
loss_fn = nn.CrossEntropyLoss()
optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)


