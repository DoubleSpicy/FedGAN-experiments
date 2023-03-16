import torch
import torchvision
import torchvision.transforms.functional as F
from PIL import Image

# read the input image
img = Image.open('flower11.jpg')

# adjust the brightness of image
img = F.adjust_brightness(img, 0.3)

# display the brightness adjusted image
img.save('test_brightness0.jpg')