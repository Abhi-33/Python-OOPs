#A custom Dataset class must implement three functions : __init__ , __len__ , and __getitem__.
#the FashionMNIST images are stored in a directory img_dir , and their labels are stored separately in a CSV file annotations_file.
import os
import pandas as pd
from torchvision.io import decode_image

class CustomImageDataset(Dataset):
    def __init__(self,annotations_file,img_dir,transform=None,target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.traget_transform = target_tranform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self,idx):
        img_path = os.path.join(self.img_dir,self.img_labels.iloc[idx,0])
        image = decode_image(img_path)
        label = self.img_labels.iloc[idx,1]
        if self.transform :
            image = self.transform(image)
        if self.target_transform :
            label = self.target_transform(label)
        return image , label
    #Display image and label.
    train_features , train_labels = next(iter(train_dataloader))
    print(f"Feature batch shape :{train_features.size()}")
    print(f"Labels batch shape : {train_labels.size()}")
    img = train_features[0].squeeze()
    label = train_labels[0]
    plt.imshow(img,cmap="gray")
    plt.show()
    print(f"Label : {label}")
