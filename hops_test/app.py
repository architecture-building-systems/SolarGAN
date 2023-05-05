# import the necessary packages
import keras.utils as image_utils
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.applications import ResNet50
import numpy as np
import cv2

import argparse
import os
from os import path
import copy
from tqdm import tqdm
import torch
from torch import nn
from gan_training import utils
from gan_training.checkpoints import CheckpointIO
from gan_training.distributions import get_ydist, get_zdist
from gan_training.eval import Evaluator
from gan_training.config import (
    load_config, build_models
)

from flask import Flask
import ghhops_server as hs

import rhino3dm

#register hops app as middleware
app = Flask(__name__)
hops = hs.Hops(app)

def load_simple_im(path):
    img =  cv2.imread(path)
    gt = np.mean(img,axis=2)/256
    gt = (gt-0.5)*2
    img_tensor = torch.from_numpy(gt).unsqueeze(0).float()
    return img_tensor

def get_one_hot(label, N):
    size = list(label.size())
    label = label.view(-1).cpu()   #reshape to a long vector
    ones = torch.sparse.torch.eye(N)
    ones = ones.index_select(0, label)   #turn to one hot
    size.append(N)  #reshape to h*w*channel classes
    return ones.view(*size).squeeze(1).permute(0,3,1,2)

def discretize_to_order_labels(t):
    tensor=t.cpu()
    bins = torch.tensor([-0.125,0.125, 0.375, 0.625, 0.875,1.125])
    inds = torch.bucketize(tensor, bins)
    tensor_discret = inds.add(-1)
    
    return tensor_discret

@hops.component(
    "/att_processing",
    name="AttProcessing",
    description="Extract attributes from processed greyscale images",
    icon="",
    inputs=[
        hs.HopsString("Path", "path", "Path to image")
    ],
    outputs=[
        hs.HopsNumber("c_mu", "c_mu", "Image attribute values"),
        hs.HopsNumber("c_var", "c_var", "Image attributes variances")
    ],
) 

def att_processing(img_path):
    configres_path = 'graycube_im_test.yaml'

    #configs
    configres = load_config(configres_path)

    c_dim = configres['dvae']['c_dim']
    out_res_name = configres['test']['out_name']

    checkpoint_res_dir = path.join(out_res_name, 'chkpts')
    batch_size = configres['test']['batch_size']

    dvae, generator_res, discriminator_res = build_models(configres)
    dvae_ckpt_path = os.path.join('outputs', configres['dvae']['runname'], 'chkpts', configres['dvae']['ckptname'])
    dvae_ckpt = torch.load(dvae_ckpt_path, map_location=torch.device('cpu'))['model_states']['net']
    dvae.load_state_dict(dvae_ckpt)

    # Put models on gpu if needed
    is_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if is_cuda else "cpu")
    dvae = dvae.to(device)

    tensor = load_simple_im(img_path)

    x_real_shift = tensor.add(1).div(2)

    if x_real_shift.size(0) == 1:
        x_real_disc = discretize_to_order_labels(x_real_shift)
        x_real_onehot = get_one_hot(x_real_disc, 5)
        x_real_onehot = x_real_onehot.to(device)

        c, c_mu, c_logvar = cs = dvae(x_real_onehot, encode_only=True)
    else:
        x_real_shift = x_real_shift.to(device)
        c, c_mu, c_logvar = cs = dvae(x_real_shift, encode_only=True)
    
    c_mu = c_mu.cpu().detach().numpy().squeeze()
    c_var = np.exp(c_logvar.cpu().detach().numpy().squeeze())
    
    return (c_mu.tolist(), c_var.tolist())


if __name__ == "__main__":
    app.run()



