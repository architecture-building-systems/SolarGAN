import argparse
import sys
import os
from os import path
import torch
from torch import nn
from gan_training.config import (
    load_config, build_models
)

from flask import Flask
import ghhops_server as hs

import cv2
import numpy as np
from PIL import Image
import tqdm

sys.path.append("..")

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

from gan_training.distributions import get_ydist, get_zdist

from vrProjector_master import vrProjector

#register hops app as middleware
app = Flask(__name__)
hops = hs.Hops(app)

dir_path = os.path.dirname(os.path.realpath(__file__))

class CheckpointIO(object):
    def __init__(self, checkpoint_dir='./chkpts', **kwargs):
        self.module_dict = kwargs
        self.checkpoint_dir = checkpoint_dir

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

    def register_modules(self, **kwargs):
        self.module_dict.update(kwargs)

    def save(self, it, filename):
        filename = os.path.join(self.checkpoint_dir, filename)

        outdict = {'it': it}
        for k, v in self.module_dict.items():
            outdict[k] = v.state_dict()
        torch.save(outdict, filename)

    def load(self, filename):
        filename = os.path.join(self.checkpoint_dir, filename)

        if os.path.exists(filename):
            print('=> Loading checkpoint...')
            out_dict = torch.load(filename,map_location=torch.device('cpu')) ##only use CPU here!!!
            it = out_dict['it']
            for k, v in self.module_dict.items():
                if k in out_dict:
                    v.load_state_dict(out_dict[k])
                else:
                    print('Warning: Could not find %s in checkpoint!' % k)
        else:
            it = -1

        return it

configres_path = os.path.join(dir_path, 'sbe_imtest_single.yaml')

#configs
configres = load_config(configres_path)

c_dim = configres['dvae']['c_dim']
out_res_name = configres['test']['out_name']
batch_size = configres['test']['batch_size']
checkpoint_res_dir = path.join(out_res_name, 'chkpts')

checkpoint_res_io = CheckpointIO(
    checkpoint_dir=checkpoint_res_dir
)

dvae, generator_res, discriminator_res = build_models(configres)
dvae_ckpt_path = os.path.join(dir_path, 'outputs', configres['dvae']['runname'], 'chkpts', configres['dvae']['ckptname'])
dvae_ckpt = torch.load(dvae_ckpt_path, map_location=torch.device('cpu'))['model_states']['net']
dvae.load_state_dict(dvae_ckpt)

# Put models on gpu if needed
is_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if is_cuda else "cpu")

dvae = dvae.to(device)
generator_res = generator_res.to(device)
discriminator_res = discriminator_res.to(device)

# Use multiple GPUs if possible
generator_res = nn.DataParallel(generator_res)
discriminator_res = nn.DataParallel(discriminator_res)

# Register modules to checkpoint
checkpoint_res_io.register_modules(
    generator=generator_res,
    discriminator=discriminator_res,
)

it_res = checkpoint_res_io.load('model_00499999.pt')

zdist = get_zdist(configres['z_dist']['type'], configres['z_dist']['dim'], device=device)

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

def generator_postprocess(x):
    n_channels = x.size(1)
    
    if n_channels ==5:
        x_im = x.add(1).div(2).clamp(0,1).round().argmax(dim=1).unsqueeze(1).cpu()/(n_channels-1)
        return x_im
        
    else:
        x_shift = x.add(1).div(2).cpu()
        x_discret = discretize_to_order_labels(x_shift)

        return x_discret.div(4)
    
def colorize_one_image(gray_image, n):
    
    def colormap(n):
        cmap=np.zeros([n, 3]).astype(np.uint8)
        cmap[0] = np.array([255,255,255])  ## 	black edge: blac
        cmap[1] = np.array([218,165,32])   ## 	ground: Goldenrod
        cmap[2] = np.array([205,92,92])    ## 	opaque surfaces: IndianRed
        cmap[3] = np.array([135,206,250])  ## 	glazing: LightSkyBlue
        cmap[4] = np.array([240,255,255	]) ## 	sky: Azure

        return cmap
    
    cmap = colormap(n)
    size = gray_image.size()  # output
    color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0) 

    for label in range(0, len(cmap)):
        mask = gray_image[0] == label  
        color_image[0][mask] = cmap[label][0] 
        color_image[1][mask] = cmap[label][1] 
        color_image[2][mask] = cmap[label][2]

    return color_image

def gen_pure_white(w,h):

        b = g =r  = np.ones((w,h), dtype=np.uint8)*255


        white = cv2.merge([b, g, r])
        
        return white
    
def cubemap_back_to_fisheye(single_im_tensor,n_channels=5, outsize=256, padding=4):
    x_1 = colorize_one_image(single_im_tensor,5)

    npimg = np.transpose(x_1.numpy(), (1,2,0))
    npimg_front = npimg[32:96,32:96,:]
    npimg_top2rotate=np.vstack([gen_pure_white(32,64), npimg[0:32,32:96,:]])
    npimg_top = np.rot90(npimg_top2rotate,1)
    npimg_bottom2rotate=np.vstack([npimg[96:128,32:96,:],gen_pure_white(32,64)])
    npimg_bottom = np.rot90(npimg_bottom2rotate,-1)
    npimg_right=np.hstack([npimg[32:96,96:128,:],gen_pure_white(64,32)])
    npimg_left=np.hstack([gen_pure_white(64,32),npimg[32:96,0:32,:]])
    npimg_back=gen_pure_white(64,64)
    
    source = vrProjector.CubemapProjection()
    #source.loadImages("cubemap_reverse/left.png","cubemap_reverse/front.png","cubemap_reverse/right.png","cubemap_reverse/back.png","cubemap_reverse/top.png","cubemap_reverse/bottom.png")
    source.readImageArrays(npimg_left,npimg_front,npimg_right,npimg_back,npimg_top,npimg_bottom)
    #source.set_use_bilinear(True)


    out = vrProjector.SideBySideFisheyeProjection()
    out.initImage((outsize-padding*2)*2,outsize-padding*2)
    out.reprojectToThis(source)
    #out.saveImage("cubemap_reverse/fisheye.png")
    x_fish_dual = out.outputImage("fisheye")
    
    x_fish = x_fish_dual[:,outsize-padding*2:(outsize-padding*2)*2,:]
    blk_mask=x_fish==0
    x_fish[blk_mask] = 255
    x_fish = np.pad(x_fish, ((padding,padding), (padding,padding),(0, 0)), 'constant', constant_values=255)


    #plt.imshow(x_fish)
    #plt.axis("off")
    
    return x_fish



@hops.component(
    "/att_processing",
    name="AttProcessing",
    description="Extract attributes from processed greyscale images",
    icon="",
    inputs=[
        hs.HopsString("Path", "path", "Path to image")
    ],
    outputs=[
        hs.HopsNumber("c_mu", "c_mu", "Image attribute values", hs.HopsParamAccess.LIST),
        hs.HopsNumber("c_var", "c_var", "Image attributes variances", hs.HopsParamAccess.LIST)
    ],
) 
def att_processing(img_path):

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
    "/wwr_traversal",
    name="WWR Traversal",
    description="Traverse latent space dimensions",
    icon="",
    inputs=[
        hs.HopsNumber("c_mu", "c_mu", "Latent image features", access=hs.HopsParamAccess.LIST),
        hs.HopsNumber("c_dim","c_dim","Feature dimension to traverse"),
        hs.HopsNumber("Value", "val", "New value for sleected dimension")
    ],
    outputs=[
        hs.HopsString("Image", "image", "New Image")
    ],
) 

def wwr_traversing(c_mu2, c_dim, val):

    c_mu2 = torch.FloatTensor(c_mu2)
    c_mu2 = c_mu2[None, :]

    c_dim = int(c_dim)

    #remove images from folder from previous run

    out_path = os.path.join(dir_path, (str(val) + ".PNG"))
    if os.path.isfile(out_path):
        try:
            os.remove(out_path)
        except OSError as e:
            print("Error: %s : %s" % (out_path, e.strerror))

    z = zdist.sample((batch_size,))

    #for i in range(1):

    #c_ = c_mu[i:i+1]
    c_ = c_mu2

    z_ = z[0:1]

    #for val in interpolation:
    c_p = c_
    c_p[:,c_dim] = val

    z_p_ = torch.cat([z_, c_p], 1)

    idganres_sample_p = generator_postprocess(generator_res(z_p_)).data.cpu().squeeze(0)
    """
        img_name = cubemap_back_to_fisheye(idganres_sample_p*4, n_channels = 5, id = ix)
    out_path = os.path.join(dir_path, img_name)
    imgs.append(out_path) """
    x_gan = Image.fromarray(cubemap_back_to_fisheye(idganres_sample_p*4, n_channels = 5))

    gan_file = os.path.join(dir_path, (str(val)+'.PNG'))
    print(gan_file)

    try:
        x_gan.save(gan_file)
    except OSError as e:
        print("Error: %s : %s" % (out_path, e.strerror))


    img_names_gan = os.path.join(dir_path, (str(val)+'.PNG'))

    return img_names_gan


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
        hs.HopsNumber("Latitude", "Lat", "Location latitude"),
        hs.HopsNumber("Longitude", "Long", "Location longitude "),
        hs.HopsNumber("Height", "height", "z-Coordinate of the sensor point"),
        hs.HopsNumber("Surface Normal X", "surfaceNormalX", "x component of the surface normal vector given the facade sensor point"),
        hs.HopsNumber("Surface Normal Y", "surfaceNormalY", "y component of the surface normal vector given the facade sensor point"),
        hs.HopsNumber("Monthly index", "monthIndex", " Monthly index of the weekly patch"),
        hs.HopsNumber("Solar Declination", "solarDecl", "z-Coordinate of the sensor point"),
        hs.HopsNumber("Weather statistics", "weatherStats", "For DNI and DHI of each week: hourly peak and hourly average; hourly average of the max./min. day"),
    ],
    outputs=[
        hs.HopsNumber("Time series", "time series", "Generated time series of solar irradiation"),
    ],
) 

def timeseries_gen(image_features, long, lat, height, surfaceNormalX,  surfaceNormalY, monthIndex, solarDecl, weatherStats):
    
    #47 attributes: lat, long, height, 32 image features, surface X, surface Y, monthIndex, declination, 8 weather 
    #attributes = np.hstack((lat, long, height, image_features, surfaceNormalX, surfaceNormalY, monthIndex, solarDecl, weatherStats))

    attributes = np.load(os.path.join(dir_path,'sbe_att.npy'))

    train_sample_size = attributes.shape[0]

    features = np.load(os.path.join(dir_path,'sbe_feat.npy'))
    features = features.reshape(-1,119,1)

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
    (data_feature, data_attribute, data_attribute_outputs, real_attribute_mask) = normalize_per_sample(data_all, data_attribut, data_feature_outputs, data_attribute_outputs)

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

    checkpoint_dir = os.path.join(dir_path,"solargan_training/results/checkpoint")
    sample_dir = os.path.join(dir_path,"solargan_training/results/sample")
    time_path = os.path.join(dir_path,"solargan_training/results/time/time.txt")

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
    #run_config = tf.compat.v1.ConfigProto()

    tf.reset_default_graph()
    #tf.compat.v1.reset_default_graph()

    sess = tf.Session(config=run_config)
    #sess = tf.compat.v1.Session(config=run_config)

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
        print("Finished Building")
        gan.load(checkpoint_dir)
        print("Finished loading")
    
    return 0

if __name__ == "__main__":
    app.run(debug=True)



