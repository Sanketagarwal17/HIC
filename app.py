"""Create an Hotel Image Classification Web App using PyTorch and Streamlit."""

# import libraries
from PIL import Image
from torchvision import models
import torch
import numpy as np
import streamlit as st
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from albumentations.pytorch import ToTensorV2
import albumentations as A


class ImgModel(nn. Module):
    def __init__(self, hidden_layer_size,num_classes):
        super().__init__()
        #self.norm1 = nn.BatchNorm2d(3)
        self.effnet = EfficientNet.from_pretrained('efficientnet-b4')
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(0.1)
        self.dense_layer1 = nn.Linear(1792, hidden_layer_size)
        #self.activation = nn.SiLU()
        self.norm1 = nn.BatchNorm1d(hidden_layer_size)
        #self.dense_layer2 = nn.Linear(hidden_layer_size , 256)
        #self.norm2 = nn.BatchNorm1d(256)
        self.out = nn.Linear(256, num_classes)

    def forward(self, inputs):
        
        #eff_out1 = self.norm1(inputs[0])
        eff_out1 = self.effnet.extract_features(inputs['images'])
        eff_out1 = nn.Flatten()(self._avg_pooling(eff_out1))
        eff_out1 = self.drop(eff_out1)
        #self.dense_layer[i].to("cuda")
        
        output = self.dense_layer1(eff_out1)
        output = self.norm1(output)
        #output[0] = self.activation(output[0])
        
        #output = self.dense_layer2(output)
        #output = self.norm2(output)
        
        output = self.out(output)
        output = nn.Softmax()(output)
        
        return output

is_cpu=True
model_path='./models/val_loss_1.8715epoch_67.pth'

model = ImgModel(256, 15)
device = "cpu"
model.to(device)
params = torch.load(model_path, map_location=torch.device('cpu'))
model.load_state_dict(params['model'])
model.eval()

Classes = ["Balcony","Bar","Bathroom","Bedroom","Bussiness Centre","Dining room","Exterior",
           "Gym","Living room","Lobby","Patio","Pool","Restaurant","Sauna","Spa"]

class_to_ind = {}
ind_to_class = {}
for i,cl in enumerate(Classes):
    class_to_ind[cl] = i
    ind_to_class[i] = cl

# set title of app
st.title("Hotel Image Classification App")
st.write("")

# enable users to upload images for the model to make predictions
file_up = st.file_uploader("Upload an image", type = "jpg")


def predict(image):
    """Return prediction with highest probability.
    Parameters
    ----------
    :param image: uploaded image
    :type image: jpg
    :rtype: list
    :return: Predicted class
    """
    # transform the input image through resizing, normalization
    transform = A.Compose([
    #A.SmallestMaxSize(max_size=256),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.5),
    #A.RandomCrop(height=256, width=256),
    A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.HorizontalFlip(),
    A.Rotate(limit=10,p=.5),
    A.Resize(256,256,interpolation = 1),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1),
    ToTensorV2(),
    ])
    # load the image, pre-process it, and make predictions
    #img = cv2.imread(image_path)
    #img =  cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #img = transform(image = img)['image']
    img = Image.open(image)
    img = np.array(img)
    img = transform(image=img)['image']
    inputs = {};
    inputs["images"] = img.unsqueeze(dim=0).to(device)
    inputs["labels"] = -1

    out = model(inputs)
    out.to('cpu')
    #print(out)
    pred = torch.max(out, 1)[1].cpu().detach().numpy()
    #print(pred)
    return ind_to_class[pred[0]]


if file_up is not None:
    # display image that user uploaded
    image = Image.open(file_up)
    col1, col2, col3 = st.columns([3,3,3])
    with col1:
        st.write("")
    with col2:
        st.image(image, caption = 'Input Image')
    with col3:
        st.write("")
    st.write("")
    label = predict(file_up)
    st.markdown("<h3 align=\"center\"><u>Prediction</u>:   "+label+"</h3>",unsafe_allow_html=True)
