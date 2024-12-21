import torch
from PIL import Image
# load model
import torchvision.transforms as transforms
model = torch.hub.load('hustvl/yolop', 'yolop', pretrained=True)

img = Image.open("YOLOP/input/DE_BBBR667_2015-04-17_10-56-05-035846_k0.jpg")
img = img.resize((640,640))

transform = transforms.Compose([
    transforms.PILToTensor()
])

# transform = transforms.PILToTensor()
# Convert the PIL image to Torch tensor
img_tensor = transform(img)

det_out, da_seg_out,ll_seg_out = model(img_tensor)

print(ll_seg_out)