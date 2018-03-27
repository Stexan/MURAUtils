import os
import numpy as np
import time
import sys
from PIL import Image

import cv2

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


#--------------------------------------------------------------------------------


def generate(pathImageFile, pathOutputFile, model, transCrop = 224):
	normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	transformList = []
	transformList.append(transforms.Resize((transCrop, transCrop)))
	transformList.append(transforms.ToTensor())
	transformList.append(normalize)      
	
	trsf = transforms.Compose(transformList)

    #---- Load image, transform, convert 
	imageData = Image.open(pathImageFile).convert('RGB')
	imageData = trsf(imageData)
	imageData = imageData.unsqueeze_(0)

	input = torch.autograd.Variable(imageData)
	
	output = model(input)
	#print(model.eval())

	weights = list(model.parameters())[-2].data.numpy()
	lastConvOut = model.lastConvOutput.data.numpy()
	lastConvOut = lastConvOut[0, :, :, :]
	#---- Generate heatmap

	print(weights)
	print(weights.shape)
	print(lastConvOut.shape)

	heatmap = np.zeros(dtype=np.float32, shape = lastConvOut.shape[1:3])
	print(heatmap.shape)
	for i in range (0, len(weights[0])):
		map = lastConvOut[i,:,:]
		heatmap += weights[0][i] * map
        
        #---- Blend original and heatmap 
	
	npHeatmap = heatmap

	imgOriginal = cv2.imread(pathImageFile, 1)
	originalWidth, originalHeight, _ = imgOriginal.shape
        
	cam = npHeatmap / np.max(npHeatmap)
	cam = cv2.resize(cam, (originalHeight, originalWidth ))
	
	finalHeatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
	finalHeatmap[np.where(cam < 0.3)] = 0

	img = cv2.addWeighted(imgOriginal.astype('uint8'), 0.8, finalHeatmap.astype('uint8'), 0.3, 0)
    
	cv2.imwrite(pathOutputFile, img)
	#cv2.imwrite(pathOutputFile.replace('.png','Heat.png'), finalHeatmap)
	return output.cpu().data.numpy()[0][0]