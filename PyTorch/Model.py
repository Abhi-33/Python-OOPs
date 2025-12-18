import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt 


training_data = datasets.FashionMNIST(
    root="data",
    train=True, 
    download=True,
    transform=ToTensor(),
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 64

train_dataloader = DataLoader(training_data , batch_size=batch_size)
test_dataloader = DataLoader(test_data ,batch_size=batch_size)

device  = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device : {device}")

class NeuralNetwork(nn.Module) :
    def __init__(self):
        super().__init__() # turns on PyTorch engine.
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(), #nn.Linear(in_features , out_features) -> learns basic pattterns ,  O/P = output*weight + bias
            nn.ReLU(), # adds non-linearity , max(0,x)
            nn.Linear(), 
            nn.ReLU(),
            nn.Linear() #logits (unnormalized scores) -> will be passed to CrossEntropyLoss  for classification.
        )
    def forward(self,x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
    
model = NeuralNetwork().to(device)
print(model)
        
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=0.01) #SGD -> Stochastic Gradient Descent
# optimizer = torch.optim.Adam(model.parameters(),lr=0.002) # Adam -> Adaptive moment estimation

# training loop
def train(dataloader , model , loss_fn , optimizer):
    size = len(dataloader.dataset) # getting the length of the dataset
    model.train() # sets the model to training mode.
    
    
    for batch , (x,y) in enumerate(dataloader):
         x,y = x.to(device) , y.to(device)
         
         optimizer.zero_grad() 
         pred = model(x) # forward pass
         loss = loss_fn(pred,y)
         
        #  backpropagation
         loss.backward() #computing gradients using backpropagation
         optimizer.step() #uses gradients > updates weights & bias -> thsi is where "learning actually happens."
         
         if batch % 100 == 0 :
             loss , current = loss.item() , (batch+1) * len(x)
             print(f"loss : {loss:>7f} [{current:>5d}/{size:>5d}] ")
             
def test(dataloader , model , loss_fn) :
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval() # it disables training behaviour , correct inference mode
    
    test_loss , correct = 0,0
    with torch.no_grad():
        for x,y in dataloader :
            x,y = x.to(device) , y.to(device)
            pred = model(x)
            test_loss+= loss_fn(pred,y).item()
            correct+=(pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss :{test_loss:>8f} \n")
    
    
epochs = 5
for i in range(epochs) :
    print(f"Epoch {i+1}\n-------------------------------")
    train(train_dataloader , model , loss_fn,optimizer)
    test(test_dataloader , model , loss_fn)
print("DONE!!!")


    
    




        
