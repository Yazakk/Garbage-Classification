from django.apps import AppConfig
import os
from django.conf import settings
import torch
from PyTorch import ResNet

class SubAppConfig(AppConfig):
    name = 'sub_app'
    model_resnet = model_path = os.path.join(settings.MODELS, 'checkpoint.pth')
    # model_r = torch.load(model_resnet, map_location=lambda storage, loc: storage)

    model_r = ResNet()
    optimizer = torch.optim.Adam(model_r.parameters(), lr=5.5e-5)
    checkpoint = torch.load(model_resnet, map_location=lambda storage, loc: storage)
    model_r.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optim_state'])
    epoch = checkpoint['epoch']
    # model_r.eval()




    # model_efficient = os.path.join(settings.MODELS, 'skin_model_final.pth')
    # model_e = torch.load(model_efficient, map_location=lambda storage, loc: storage)
