from pytorch2caffe import pytorch2caffe, plot_graph
import torch
from torch.autograd import Variable
import torchvision
import os
import numpy as np
import UNet

def main():
    pytorch_model = UNet.UNet(1,1, start_filts=16, depth=2)
    saved_params = torch.load('../unet_2/t7/epoch_89.pth')
    pytorch_model.load_state_dict(saved_params['state_dict'])

    test_image = np.load('../pytorch_keras/test.npy').astype(np.float32)
    x = Variable(torch.from_numpy(test_image[np.newaxis,np.newaxis,12:,:-14]))

    out = pytorch_model(x)
    pytorch2caffe(x, out, 'pillnet.prototxt','pillnet.caffemodel')

if __name__ == '__main__':
    main()
