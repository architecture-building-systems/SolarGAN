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


# dim_list = [12,16,30]
# dim_ix = 2
# ncol=3
# lim_d = 0.64
# lim = -1.5
# trav_dim = dim_list[dim_ix]
# wwr_traversing_gif('wwr_30',x_real_test_batch, c_dim_wwr = trav_dim, lim_d = lim_d, lim=lim, ncol=12,fps=3)

@hops.component(
    "/wwr_traversal",
    name="WWR Traversal",
    description="Traverse latent space dimensions",
    icon="",
    inputs=[
        hs.HopsNumber("C_mu", "c_mu", "image fature vector"),
        hs.HopsNumber("c_dim","c_dim","Feature dimension to traverse")
    ],
    outputs=[
        hs.HopsNumber("Image", "image", "New Image")
    ],
) 

def wwr_traversing(c_mu, c_dim = 30):
    
    lim = 1
    lim_d=-2.6
    ncol=4

    interpolation = torch.linspace(lim_d, lim, ncol)

    idganres_samples_p = []
    dvae_samples_p = []

    z = zdist.sample((batch_size,))

    for i in range(x_real_shift.size(0)):

        c_ = c_mu[i:i+1]
        z_ = z[i:i+1]
        c_zero = torch.zeros_like(c_)

        for val in interpolation:
            c_p = c_
            c_p[:, c_dim] = val

            #c_zero[:, c_dim_wwr] = val
            #c_p = c_ + c_zero
            z_p_ = torch.cat([z_, c_p], 1)

            idganres_sample_p = generator_postprocess(generator_res(z_p_)).data.cpu()
            idganres_samples_p.append(idganres_sample_p)

            dvae_sample_p = decoder_postprocess(dvae(c=c_p, decode_only=True)).data.cpu()
            dvae_samples_p.append(dvae_sample_p)

            x_gan = Image.fromarray(cubemap_back_to_fisheye(idganres_sample_p*4, n_channels = 5))
            gan_file = target_folder+str(ix)+'.PNG'
            x_gan.save(gan_file)


    idganres_samples_p = torch.cat(idganres_samples_p, dim=0)

    dvae_samples_p = torch.cat(dvae_samples_p, dim=0)

    return x_gan




def normalize_attribute(data_att, data_att_outputs, data_att_min, data_att_max):
    data_att_norm = data_att
    total_dim = 0
    for output in data_att_outputs:
        if output.type_ == OutputType.CONTINUOUS:
            for _ in range(output.dim):
                data_att_norm[:, total_dim] = (data_att_norm[:, total_dim] - data_att_min[total_dim]) / (data_att_max[total_dim] - data_att_min[total_dim])
                if output.normalization == Normalization.MINUSONE_ONE:
                    data_att_norm[:, total_dim] = data_att_norm[:, total_dim] * 2.0 - 1.0

                total_dim += 1
        else:
            total_dim += output.dim


    return data_att_norm

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

    gen_flags = np.ones((train_sample_size,119))

    data_feature_outputs = [
	    output.Output(type_=OutputType.CONTINUOUS,dim=1,normalization=Normalization.ZERO_ONE,is_gen_flag=False)]
        #hourly solar radiation

    data_attribute_outputs = [
        output.Output(type_=OutputType.CONTINUOUS,dim=1,normalization=Normalization.MINUSONE_ONE,is_gen_flag=False),
        #lat
        output.Output(type_=OutputType.CONTINUOUS,dim=1,normalization=Normalization.MINUSONE_ONE,is_gen_flag=False),
        #longi
        output.Output(type_=OutputType.CONTINUOUS,dim=1,normalization=Normalization.ZERO_ONE,is_gen_flag=False),
        #height
        output.Output(type_=OutputType.CONTINUOUS,dim=2,normalization=Normalization.MINUSONE_ONE,is_gen_flag=False),
        #norm vector (xy)
        output.Output(type_=OutputType.CONTINUOUS,dim=32,normalization=Normalization.MINUSONE_ONE,is_gen_flag=False),
        #latent_im
        output.Output(type_=OutputType.CONTINUOUS,dim=1,normalization=Normalization.ZERO_ONE,is_gen_flag=False),
        #mon
        output.Output(type_=OutputType.CONTINUOUS,dim=1,normalization=Normalization.MINUSONE_ONE,is_gen_flag=False),
        #inc
        output.Output(type_=OutputType.CONTINUOUS,dim=8,normalization=Normalization.ZERO_ONE,is_gen_flag=False)]
        #weather_stat

    #necessary inputs
    data_all = features
    data_attribut = attributes
    data_gen_flag = gen_flags

    sample_len = 17

    # normalise data
    (data_feature, data_attribute, data_attribute_outputs,
    real_attribute_mask) = normalize_per_sample(
            data_all, data_attribut, data_feature_outputs,
            data_attribute_outputs)

    # add generation flag to features
    data_feature, data_feature_outputs = add_gen_flag(
        data_feature, data_gen_flag, data_feature_outputs, sample_len)

    data_attribute_min = np.amin(data_attribute, axis=0)
    data_attribute_max = np.amax(data_attribute, axis=0)

    data_attribute_normlized = normalize_attribute(data_attribute, data_attribute_outputs, data_attribute_min, data_attribute_max)


    generator = DoppelGANgerGenerator(
        feed_back=True,
        noise=True,
        feature_outputs=data_feature_outputs,
        attribute_outputs=data_attribute_outputs,
        real_attribute_mask=real_attribute_mask,
        attribute_num_units =100,
        sample_len=sample_len,
        feature_num_units=100,
        feature_num_layers=2)

    discriminator = Discriminator(num_units=100)
    attr_discriminator = AttrDiscriminator(num_units=100)

    checkpoint_dir = "solargan_training/results/checkpoint"
    sample_dir = "solargan_training/results/sample"
    time_path = "solargan_training/results/time/time.txt"
    epoch = 200
    batch_size = 100
    g_lr = 0.0001
    d_lr = 0.0001 
    vis_freq = 1000
    vis_num_sample = 1
    d_rounds = 3
    g_rounds = 1
    d_gp_coe = 10.0
    attr_d_gp_coe=10.0
    attr_d_lr = 0.0001
    g_attr_d_coe = 1.0
    extra_checkpoint_freq = 1000
    num_packing = 1


    # config
    run_config = tf.ConfigProto()
    tf.reset_default_graph()


    sess = tf.Session(config=run_config)

    with sess.as_default() as sess:
        assert tf.get_default_session() is sess
        gan = DoppelGANger(
            sess=sess, 
            checkpoint_dir=checkpoint_dir,
            sample_dir=sample_dir,
            time_path=time_path,
            epoch=epoch,
            batch_size=batch_size,
            data_feature=data_feature,
            data_attribute=data_attribute,
            real_attribute_mask=real_attribute_mask,
            data_gen_flag=data_gen_flag,
            sample_len=sample_len,
            data_feature_outputs=data_feature_outputs,
            data_attribute_outputs=data_attribute_outputs,
            vis_freq=vis_freq,
            vis_num_sample=vis_num_sample,
            generator=generator,
            discriminator=discriminator,
            attr_discriminator=attr_discriminator,
            d_gp_coe=d_gp_coe,
            attr_d_gp_coe=attr_d_gp_coe,
            g_attr_d_coe=g_attr_d_coe,
            d_rounds=d_rounds,
            g_rounds=g_rounds,
            g_lr=g_lr,
            d_lr=d_lr,
            attr_d_lr = attr_d_lr,
            num_packing=num_packing,
            extra_checkpoint_freq=extra_checkpoint_freq)

        gan.build()

        gan.load(checkpoint_dir)
    
    return 0

if __name__ == "__main__":
    app.run()



