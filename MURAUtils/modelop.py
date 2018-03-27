import sys
import os
import argparse
from torchvision import transforms
from torchvision.datasets.folder import pil_loader

import torch
from torch.autograd import Variable

def load(modelPath, model):
	
	model = model.cpu()
	checkpoint = torch.load(modelPath, map_location='cpu')

	from collections import OrderedDict
	new_state_dict = OrderedDict()
	for k, v in checkpoint.items():
		name = k[7:]
		new_state_dict[name] = v

	model.load_state_dict(new_state_dict)
	model.eval()

	return model


def predict(imagePath, model):
	image = pil_loader(imagePath)
	trsf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

	image = trsf(image)
	image = image.unsqueeze(0)
	imageVar = Variable(image)

	prediction = model(imageVar)
	prediction.data.cpu().numpy()[0]

	return prediction