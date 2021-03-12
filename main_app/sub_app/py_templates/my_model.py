import os
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from ..apps import SubAppConfig

image_path=os.path.dirname(os.path.dirname(__file__))

aug = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])




def image_pred(url):
    try:
        new_url=image_path+url
    except TypeError:
        new_url=url
    img = Image.open(new_url)
    img = img.convert(mode='RGB')
    image = aug(img)
    image = image.unsqueeze(0).cpu()  # add another dimension at 0
    SubAppConfig.model_r.eval()



    out = SubAppConfig.model_r(image)

    out = torch.mean(out, dim=0)


    out = out.detach().numpy()
    out = np.exp(out) / sum(np.exp(out))

    out=np.argmax(out)
    return out.round(3)