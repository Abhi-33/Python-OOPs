class Model(nn.module):
    def __init__(self):
        super().__init__()
        #define layers

    def forward(self,x):
        #return data flow
        return output

model = Model()
loss_fn= ...
optimizer = ...

for epoch in epochs:
    for batch in dataloader :
        optimizer.zero_grad()
        output = model(x)
        loss = loss_fn(output,y)
        loss.backward()
        optimizer.step()
