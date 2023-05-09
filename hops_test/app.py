import argparse
import sys
import os
from os import path
from tqdm import tqdm
import torch
from gan_training.config import (
    load_config, build_models
)

from flask import Flask
import ghhops_server as hs
import rhino3dm

import cv2
import numpy as np
import pickle
import pandas as pd

sys.path.append("..")
import matplotlib.pyplot as plt

from gan import output
sys.modules["output"] = output

from gan.doppelganger import DoppelGANger
from gan.load_data import load_data
from gan.network import DoppelGANgerGenerator, Discriminator, AttrDiscriminator
from gan.output import Output, OutputType, Normalization
import tensorflow as tf
from gan.network import DoppelGANgerGenerator, Discriminator, \
    RNNInitialStateType, AttrDiscriminator
from gan.util import add_gen_flag, normalize_per_sample

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

@hops.component(
    "/timeseries_gen",
    name="TimeSeriesGeneration",
    description="Generate time series of solar irradiation",
    icon="",
    inputs=[
        hs.HopsNumber("Image features", "image features", "Vector of image features"),
        hs.HopsNumber("Longitude", "Long", "Location longitude "),
        hs.HopsNumber("Latitude", "Lat", "Location latitude"),
        hs.HopsNumber("Height", "height", "z-Coordinate of the sensor point"),
        hs.HopsNumber("Surface Normal", "surfaceNormal", "x, y component of the surface normal vector given the facade sensor point"),
        hs.HopsNumber("Monthly index", "monthIndex", " Monthly index of the weekly patch"),
        hs.HopsNumber("Solar Declination", "solarDecl", "z-Coordinate of the sensor point"),
        hs.HopsNumber("Weather statistics", "weatherStats", "For DNI and DHI of each week: hourly peak and hourly average; hourly average of the max./min. day"),
    ],
    outputs=[
        hs.HopsNumber("Time series", "time series", "Generated time series of solar irradiation"),
    ],
) 

def timeseries_gen(features, long, lat, height, surfaceNormal, monthIndex, solarDecl, weatherStats):
   return 0

if __name__ == "__main__":
    app.run()



