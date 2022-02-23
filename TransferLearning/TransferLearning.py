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
#device = torch.device('cpu')
device = torch.device(dev) 

def tmpfunc(x):
    return x.repeat(3, 1, 1)
transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Resize((224,224)),
    transforms.Lambda(tmpfunc)])
    #transforms.Grayscale(3)])
     #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batchSize = 128

trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

indicies = []
imagesPerCatagory = 90
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
trainloader = torch.utils.data.DataLoader(subset, batch_size=batchSize, shuffle=True, num_workers=2)

testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batchSize, shuffle=False, num_workers=2)

resnet = False
if resnet:
    model = models.resnet50(pretrained=True)
    print(model)
    model.f = nn.Linear(in_features=2048, out_features=10, bias=True)
    print(model)
else:
    model = models.vgg19(pretrained=True)
    print(model)
    model.classifier[6] = nn.Linear(in_features=4096, out_features=10, bias=True)
    print(model)

#print(model)
model = model.to(device)

# Freeze all weights but the fully connected layer
for i, (name, param) in enumerate(model.named_parameters()):
    if name != 'fc.weight' and name != 'fc.bias':
        param.requires_grad = False

epochs  = 20
lr      = 0.01
#loss_fn = nn.MSELoss()
loss_fn = nn.CrossEntropyLoss()
optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

if __name__ == '__main__':

    trainAccPerEpoch = []
    testAccPerEpoch = []
    divergence = 0
    for e in range(epochs):
        def run(loader, train):
            numCorrect = 0
            runningLoss = 0.0
            for i, (features, targets) in enumerate(loader, 0):
    
                features    = features.to(device)
                targets     = targets.to(device)


                if dev == 'cpu':
                    x = features.numpy()

                #features = torch.stack([features,features,features], 0)
                
                pred = model(features)
                
                if dev == 'cpu':
                    p = pred.detach().numpy()
                    t = targets.numpy()

                # Calculate number of correct values
                maxVals, maxIdx = torch.max(pred, 1)
                numCorrect += (maxIdx == targets).sum().item()

                if train:
                    optim.zero_grad()
                    #if type(loss_fn).__name__ == 'MSELoss':
                    #    targets = torch.nn.functional.one_hot(targets, num_classes=len(classes))
                    #    targets = targets.float()
                    
                    loss = loss_fn(pred, targets)
                    if i % 100 == 99:
                        print(f'[{e + 1}, {i + 1:5d}] loss: {runningLoss / 100:.3f}')
                        runningLoss = 0.0
                    loss.backward()
                    optim.step()
                    runningLoss += loss.item()

            return numCorrect
            
        trainAcc = run(trainloader, train=True)
        testAcc = run(testloader, train=False)

        trainAcc /= len(subset)
        testAcc /= len(testset)

        print(f'[Train,Test] accuracy for Epoch {e+1}: {[trainAcc, testAcc]}')


        trainAccPerEpoch.append(trainAcc)
        testAccPerEpoch.append(testAcc)

        if len(testAccPerEpoch) > 2 and testAccPerEpoch[-2] >= testAccPerEpoch[-1]:
            divergence += 1
            if divergence > 3:
                break
        else:
            divergence = 0



    plt.plot(range(len(trainAccPerEpoch)), trainAccPerEpoch, c='g')
    plt.plot(range(len(testAccPerEpoch)), testAccPerEpoch, c='r')
    plt.title(f'Accuracy vs. Epoch, loss = {type(loss_fn).__name__}, lr={lr}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Training Acc', 'Test Acc'])
    plt.show()
