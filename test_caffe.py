import cv2 
import numpy as np
import matplotlib.pyplot as plt
import UNet
import torch
from torch.autograd import Variable

MEAN=16.861
STD=56.475

im = cv2.imread('../pytorch_keras/test.png', cv2.IMREAD_GRAYSCALE)
im = im[12:, :-14]
blob = cv2.dnn.blobFromImage(im, 1, (480, 640))
blob = (blob - MEAN) / STD
net = cv2.dnn.readNetFromCaffe('./pillnet.prototxt', './pillnet.caffemodel')
net.setInput(blob)
out = net.forward().squeeze()
out = 1/(1 + np.exp(-out))

pytorch_model = UNet.UNet(1, 1, depth=2, start_filts=16)
saved_params = torch.load('../unet_2/t7/epoch_89.pth')
pytorch_model.load_state_dict(saved_params['state_dict'])
x = Variable(torch.from_numpy(blob))
out_pytorch = pytorch_model(x).data.numpy().squeeze()
out_pytorch = 1 / (1 + np.exp(-out_pytorch))

plt.subplot(1,3,1)
plt.imshow(out.squeeze(), cmap='gray')
plt.clim(0, 1)
plt.subplot(1,3,2)
plt.imshow(out_pytorch.squeeze(), cmap='gray')
plt.clim(0, 1)
plt.subplot(1,3,3)
plt.imshow(np.abs(out - out_pytorch))
plt.colorbar()
plt.show()






