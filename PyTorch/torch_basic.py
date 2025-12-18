import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets #the torchvision.datasets module contains Dataset objects for many real-world vision data like CIFAR, COCO etc.
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

#Every TorchVision Dataset includes two arguments : transform and target_transform to modify the samples and labels respectively.

# Download training data from open datasets.
#training_data is a Dataset object.
training_data = datasets.FashionMNIST(
    root = "data", #store(or look for) the dataset inside a folder called data.
    train = True,  #Give me the training split of the dataset
    download = True , #if dataset files are NOT found in root , download them .
    transform = ToTensor(), #Whenver I load an image , convert it into PyTorch tensor.
)

# Download Test data from open datasets.
test_data = datasets.FashionMNIST(
    root = "data",
    train = False,
    download = True,
    transform = ToTensor(),
)

batch_size = 64

# create data loaders.
train_dataloader = DataLoader(training_data ,  batch_size=batch_size)
test_dataloader = DataLoader(test_data , batch_size = batch_size)

#PIPELINE : Dataset -> DataLoader -> Batches -> Model
for x,y in test_dataloader :
     print(f"Shape of x [N,C,H,W]: {x.shape}")
     print(f"Shape of y : {y.shape} {y.dtype}")
     break
#output :
# Shape of X [N,C,H,W]: torch.Size([64, 1, 28, 28]) # it means we gave the computer 64 black and white images , each of size 28*28.
#N = batch size
#C = Channels(1= grayscale , 3= RGB)
#H , W = height of image , width of image
#[N,C,H,W] -> It is the "STANDARD IMAGE TENSOR FORMAT" in PyTorch.
# Shape of y : torch.Size([64]) torch.int64  # it means we also gave 64 answers(labels) , one for each image.
# Creating Models

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Cycle we should know
# Forward Pass -> calculate loss
# Backward pass -> calculate gradients
# Optimizer -> update weights
# Above cycle repeats for every batch

# In production we only run forward pass for inference , backward pass is used only during training.
#Forward pass makes predictions.
#Backward pass fixes mistakes.


#For images there are 2 typical families :
#     MLP/Fully Connected -> what we have here(Flatten + Linear Layers)
#     CNNs(Conv2D) -> more powerful for images.
# Here we're using MLP for understanding simplicity.

# ************ DEFINE MODEL ****************
class NeuralNetwork(nn.Module) :
    #nn.Module is the base class for all neural networks in PyTorch. -> basically it provides all the functionalities required for building , training, and deploying neural networks in PyTorch.
    #If a class inherits from nn.Module, PyTorch knows it is a model.
    def __init__(self): #constructor runs when model object is created + is used to define layers
        super().__init__() # it turns on PyTorch engine.
        self.flatten = nn.Flatten() #converts the multidimensional array into one-dimensional because Linear Layers expect 1D feature Vectors, not 2D images.
        self.linear_relu_stack = nn.Sequential( # nn.Sequential is a container that chains layers in order for better code readability
            #stack of hidden layers
            nn.Linear(28*28,512), #learns basic patterns
            nn.ReLU(),
            nn.Linear(512,512), #learns combination of patterns
            nn.ReLU(),
            nn.Linear(512,10) #output layer -> 10 classes
        )
#nn.Linear(in_features , out_features)
# output = input * weights + bias -> this is the output of ONE nn.Linear() layer only. nn.Linear produces an output using this formula.
#nn.ReLU- introduces non-linearity , allowing the network to learn complex patterns.
#nn.ReLU = modifies the output given by nn.Linear()
#ReLU(x) = max(0,x)
# next nn.Linear() again applies its own input * weights + bias.
#output from the last linear dense layer is called "logits"(unnormalized scores). It is typically pass to CrossEntropyLoss , which applies Softmax internally.
# Stacking Linear and ReLU layers allows hierarchial feature learning. The final layer outputs raw class scores , which are passed to CrossEntropyLoss for classification.

    def forward(self,x):  #this method defines how data flows through our network.
         x = self.flatten(x) #Converts image tensors [N,1,28,28] -> [N,784]
         logits = self.linear_relu_stack(x) # pass flattened input through the chain of layers. : Linear -> ReLU -> Linear -> ReLU -> Linear
         return logits # returns outputs of shape[N,10] - one score per class.
# we don't call forward() usually , we just do model(x) , and PyTorch calls forward internally.

# Creating the model and moving it to device
#NeuralNetwork() -> creates an instance of our class.
# .to(device) -> moves the model parameters to "cuda" or "cpu". so if "cuda" all weights are stored on GPU memory.
model = NeuralNetwork().to(device)
print(model) # prints a summary of our model architecture : Flatten , Sequential with Linear/ReLU layers.

# To train a model , we need a "loss function" and an "optimizer".
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=1e-3)

# training loop
#this function does ONE JOB : Train the model for one full pass over the dataset(one epoch).
def train(dataloader,model,loss_fn,optimizer) :
    """
    The training loop iterates over batches, moves data to the correct device,
     performs a forward pass to get predictions, computes loss, backpropagates gradients,
    updates model parameters using the optimizer, and clears gradients to prevent accumulation.
    This process repeats for all batches in one epoch.
    """
    size = len(dataloader.dataset) # here we are getting the length of the dataset
    model.train() # sets the model to training mode.

    #  dataloader gives one batch at a time.
    # X -> batch of images [batch_size ,1 , 28 ,28]
    # y -> batch of labels [batch_size]
    # batch -> batch index(0,1,2...)
    for batch , (X,y) in enumerate(dataloader) :
       X, y = X.to(device), y.to(device) # Moves data to CPU or GPU : Both model and tensors must be on the same device.

        # Compute prediction error
       optimizer.zero_grad() # it is used because gradients gets accumulated by default in PyTorch without clearing they would add up which will lead to wrong learning. gradients are used to know how much a weight contributes to a error.
       pred = model(X) # forward pass : call forward(X) -> produces logits -> Shape:[batch_size,num_classes] , no learning yet - jsut prediction.
       loss = loss_fn(pred,y) # tells How wrong are predictions ? : Compares predicted logits with true labels and produces a single scalar loss value.


        # Backpropagation
       loss.backward() #backward computes gradients usign backpropagation.
       optimizer.step() # uses gradients > updates weights & bias -> this is where "learning actually happens."


       if batch % 100 == 0 :
           loss,current = loss.item(), (batch+1) * len(X)
           print(f"loss : {loss:>7f} [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    """
    The training process is conducted over several iterations (epochs).
    During each epoch,
    the model learns parameters to make better predictions.
    """
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


epochs = 5  
#epoch -> one complete pass through the entire training dataset.    
for i in range(epochs) :
    print(f"Epoch {i+1}\n-----------------------------------")
    train(train_dataloader,model,loss_fn,optimizer)
    test(test_dataloader,model,loss_fn)
print("Done!")

# Saving models
# A Common way to save a model is to serialize the internal state dictionary (containing the model parameters).
torch.save(model.state_dict(),"model.pth")
print("Saved PyTorch Model state to model.pth")

def main():
    epochs  = 5
    for i in range(epochs) :
        print("Epoch {i+1}\n--------------------------------------")
        train(train_dataloader , model , loss_fn , optimizer)
        test(test_dataloader , model ,loss_fn)

    torch.save(model.state_dict(),"model.pth")
    print("Saved PyTorch Model state to model.pth")


if __name__ == "__main__":
    main()