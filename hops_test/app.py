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

previous_result = []

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

def renormalize_per_sample(data_feature, data_attribute, data_feature_outputs,
                           data_attribute_outputs, gen_flags,
                           num_real_attribute):
    attr_dim = 0
    for i in range(num_real_attribute):
        attr_dim += data_attribute_outputs[i].dim
    attr_dim_cp = attr_dim

    fea_dim = 0
    for output in data_feature_outputs:
        if output.type_ == OutputType.CONTINUOUS:
            for _ in range(output.dim):
                max_plus_min_d_2 = data_attribute[:, attr_dim]
                max_minus_min_d_2 = data_attribute[:, attr_dim + 1]
                attr_dim += 2

                max_ = max_plus_min_d_2 + max_minus_min_d_2
                min_ = np.zeros_like(max_plus_min_d_2 - max_minus_min_d_2)

                max_ = np.expand_dims(max_, axis=1)
                min_ = np.expand_dims(min_, axis=1)

                if output.normalization == Normalization.MINUSONE_ONE:
                    data_feature[:, :, fea_dim] = \
                        (data_feature[:, :, fea_dim] + 1.0) / 2.0

                data_feature[:, :, fea_dim] = \
                    data_feature[:, :, fea_dim] * (max_ - min_) + min_

                fea_dim += 1
        else:
            fea_dim += output.dim

    tmp_gen_flags = np.expand_dims(gen_flags, axis=2)
    data_feature = data_feature * tmp_gen_flags

    data_attribute = data_attribute[:, 0: attr_dim_cp]

    return data_feature, data_attribute

def gen_annual(gt_feat_all,given_att_all,list_index,num_gen,location_index,renorm_factor, gan, sample_len, data_feature_outputs, data_attribute_outputs, canvas_attribute):
    print(list_index)
    num_weeks = 52
    #print(given_att_all.shape)
    
    train_sample_size = given_att_all.shape[0]
    index = 3*list_index*num_weeks+int(location_index*train_sample_size/5)
    # given_att = given_att_all[index:index+num_weeks,:-2]
    # given_att_tiles=np.tile(given_att,(num_gen,1))
    #print("GIVEN_ATT")
    #print(given_att.shape)
    #print(given_att_tiles.shape)

    #tiling to repeat for the number of desired samples (and cut of the last two entries of each vector, these are added during normalization, we don't need them here)
    canvas_att_tiles=np.tile(canvas_attribute[:,:-2],(num_gen,1))
    print("NORMALIZED CANVAS ATTRIBUTE SHAPE AFTER TILING")
    print(canvas_att_tiles.shape)
    
    length = int(gt_feat_all.shape[1] / sample_len)
    num_real_attribute=8
    
    addi_attribute_input_noise = gan.gen_attribute_input_noise(
                num_gen*num_weeks)
    feature_input_noise = gan.gen_feature_input_noise(
                num_gen*num_weeks, length)

    input_data = gan.gen_feature_input_data_free(
                num_gen*num_weeks)
    
    # generate features, attributes and lengths
    gen_features, gen_attributes, gen_flags, lengths = gan.sample_from(
        None, addi_attribute_input_noise,
        feature_input_noise, input_data, given_attribute=canvas_att_tiles)

    #denormalise accordingly
    gen_features, gen_attributes = renormalize_per_sample(
        gen_features, gen_attributes, data_feature_outputs,
        data_attribute_outputs, gen_flags,
        num_real_attribute=num_real_attribute)
    
    
    gt = gt_feat_all[index:index+num_weeks,:]
    gt = gt.reshape(1,-1,1)

    gen_features=gen_features*renorm_factor
    gen_features=gen_features.squeeze(2).reshape(num_gen,-1)
    gen_features = gen_features.transpose()
    gen_features = np.expand_dims(gen_features,0)
    
    return gt, gen_features


def all_annual_batch (gt_feat_all,given_att_all,num,num_gen,site_i,renorm_factor, gan, sample_len, data_feature_outputs, data_attribute_outputs,canvas_attribute):
    weather_data_array_morning_batch = np.zeros((364,4,num_gen))
    weather_data_array_morning_single = np.zeros((364,4,1))
    weather_data_array_evening_batch = np.zeros((364,3,num_gen))
    weather_data_array_evening_single = np.zeros((364,3,1))
    
    feat_gt_all = np.empty(shape=(0,8736,1))
    feat_gen_all = np.empty(shape=(0,8736,num_gen))
    
    for i in range(num):
        
        #gt_i, gen_i = gen_annual(gt_feat_all,given_att_all,i,num_gen,site_i,renorm_factor, gan, sample_len, data_feature_outputs, data_attribute_outputs)
        gt_i, gen_i = gen_annual(gt_feat_all,given_att_all,i,num_gen,site_i,renorm_factor, gan, sample_len, data_feature_outputs, data_attribute_outputs, canvas_attribute)
        gt_i_daily = gt_i.reshape(364,-1,1)
        gen_i_daily = gen_i.reshape(364,-1,num_gen)
        
        gt_i_daily_complete = np.concatenate((weather_data_array_morning_single,gt_i_daily,weather_data_array_evening_single),axis=1)
        gen_i_daily_complete = np.concatenate((weather_data_array_morning_batch,gen_i_daily,weather_data_array_evening_batch),axis=1)
        
        gt_i_complete = np.expand_dims(gt_i_daily_complete.reshape(-1,1),0)
        gen_i_complete = np.expand_dims(gen_i_daily_complete.reshape(-1,num_gen),0)
        
        feat_gt_all = np.append(feat_gt_all,gt_i_complete).reshape(-1,8736,1)
        feat_gen_all = np.append(feat_gen_all,gen_i_complete).reshape(-1,8736,num_gen)
        
    return feat_gt_all, feat_gen_all

def calculate_weather_stats(epw_path):

    epw = open(epw_path, 'r')
    lines = epw.readlines()
    hours = lines[8:]

    DNI_weekly_peak_list = []
    DNI_weekly_avg_list = []
    DHI_weekly_peak_list = []
    DHI_weekly_avg_list = []

    DNI_max_avg_list = []
    DNI_min_avg_list = []
    DHI_max_avg_list = []
    DHI_min_avg_list = []

    #parse by week
    for w in range(52):
        DNI_weekly_average = 0
        DNI_weekly_peak = 0

        DHI_weekly_average = 0
        DHI_weekly_peak = 0

        dni_total = 0
        dhi_total = 0
        dni_day_total = 0
        dhi_day_total = 0

        DNI_day_avg = []
        DHI_day_avg = []

        hour_count = 0

        week_hours = hours[w*168:w*168+168]
        
        for h in week_hours:
            values = h.split(",")
            dni = int(values[14])
            dhi = int(values[15])

            dni_total += dni
            dhi_total += dhi
            dni_day_total += dni
            dhi_day_total += dhi

            DNI_weekly_peak = dni if DNI_weekly_peak < dni else DNI_weekly_peak
            DHI_weekly_peak = dhi if DHI_weekly_peak < dhi else DHI_weekly_peak

            hour_count += 1
            if (hour_count == 24):
                hour_count = 0
                DNI_day_avg.append(dni_day_total/24)
                DHI_day_avg.append(dhi_day_total/24)
                dni_day_total = 0
                dhi_day_total = 0

        DNI_weekly_average = dni_total / 168
        DHI_weekly_average = dhi_total / 168

        DNI_max_avg_list.append(max(DNI_day_avg))
        DNI_min_avg_list.append(min(DNI_day_avg))
        DHI_max_avg_list.append(max(DHI_day_avg))
        DHI_min_avg_list.append(min(DHI_day_avg))

        DNI_weekly_avg_list.append(DNI_weekly_average)
        DNI_weekly_peak_list.append(DNI_weekly_peak)
        DHI_weekly_avg_list.append(DHI_weekly_average)
        DHI_weekly_peak_list.append(DHI_weekly_peak)
    
    return DNI_weekly_avg_list, DNI_weekly_peak_list, DHI_weekly_avg_list, DHI_weekly_peak_list, DNI_max_avg_list, DNI_min_avg_list, DHI_max_avg_list, DHI_min_avg_list


@hops.component(
    "/timeseries_gen",
    name="TimeSeriesGeneration",
    description="Generate time series of solar irradiation",
    icon="",
    inputs=[
        hs.HopsBoolean("Run", "run", "Run time series gen", hs.HopsParamAccess.ITEM),
        hs.HopsNumber("Ensemble size", "num_gen", "Number of generated time series", hs.HopsParamAccess.ITEM),
        hs.HopsNumber("Image features", "image features", "Vector of image features", hs.HopsParamAccess.LIST),
        hs.HopsNumber("Latitude", "lat", "Location latitude", hs.HopsParamAccess.ITEM),
        hs.HopsNumber("Longitude", "long", "Location longitude", hs.HopsParamAccess.ITEM),
        hs.HopsNumber("Height", "height", "z-Coordinate of the sensor point", hs.HopsParamAccess.ITEM),
        hs.HopsNumber("Surface Normal X", "surfaceNormalX", "x component of the surface normal vector given the facade sensor point",hs.HopsParamAccess.ITEM),
        hs.HopsNumber("Surface Normal Y", "surfaceNormalY", "y component of the surface normal vector given the facade sensor point",hs.HopsParamAccess.ITEM),
        #hs.HopsNumber("Monthly index", "monthIndex", " Monthly index of the weekly patch",hs.HopsParamAccess.ITEM),
        hs.HopsNumber("Solar Declination", "solarDecl", "Declination of the sun",hs.HopsParamAccess.LIST),
        hs.HopsString("EPW File Path", "EPWfilePath", "Path to EPW file",hs.HopsParamAccess.ITEM),
    ],
    outputs=[
        hs.HopsNumber("Time series", "time series", "Generated time series of solar irradiation", hs.HopsParamAccess.LIST),
        hs.HopsNumber("DNI weekly average", "DNI weekly average", "", hs.HopsParamAccess.LIST),
        hs.HopsNumber("DNI weekly peak", "DNI weekly peak", "", hs.HopsParamAccess.LIST),
        hs.HopsNumber("DHI weekly average", "DHI weekly average", "", hs.HopsParamAccess.LIST),
        hs.HopsNumber("DHI weekly peak", "DHI weekly peak", "", hs.HopsParamAccess.LIST),
        hs.HopsNumber("DNI weekly max avg", "DNI weekly max avg", "", hs.HopsParamAccess.LIST),
        hs.HopsNumber("DNI weekly min avg", "DNI weekly min avg", "", hs.HopsParamAccess.LIST),
        hs.HopsNumber("DHI weekly max avg", "DHI weekly max avg", "", hs.HopsParamAccess.LIST),
        hs.HopsNumber("DHI weekly min avg", "DHI weekly min avg", "", hs.HopsParamAccess.LIST),
    ],
) 

def timeseries_gen(run, num_gen, img_feat, lat, long, height, x_norm, y_norm, solar_dec, epw_path):
    global previous_result 
    
    if (run == 0):
        return previous_result

    num_gen = int(num_gen)
    img_feat = list(img_feat)
    solar_dec = list(solar_dec)

    DNI_weekly_avg_list, DNI_weekly_peak_list, DHI_weekly_avg_list, DHI_weekly_peak_list, DNI_max_avg_list, DNI_min_avg_list, DHI_max_avg_list, DHI_min_avg_list = calculate_weather_stats(epw_path)
    weather_stat = np.vstack((DNI_weekly_peak_list, DHI_weekly_peak_list, DNI_weekly_avg_list, DHI_weekly_avg_list, DNI_max_avg_list, DHI_max_avg_list, DNI_min_avg_list, DHI_min_avg_list))

    month_index = [1,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7,7,8,8,8,8,9,9,9,9,10,10,10,10,10,11,11,11,11,12,12,12,12]

    canvas_attribute_stack = []
    #slice the weekly weather stats apart, stick them into their respective weekly attribute vector, then stack the attr vectors on top of each other
    for i in range(52):
        #47 (49) attributes: lat, long, height, 32 image features, surface X, surface Y, monthIndex, declination, 8 weather, two filler numbers to match the dimension to the processed attributes from the .npy file
        one_week_attr = np.hstack((lat, long, height, img_feat, x_norm, y_norm, month_index[i], solar_dec[i], weather_stat[:,i], 1, 1))
        if (i == 0):
            canvas_attribute_stack = one_week_attr
        else:
            canvas_attribute_stack = np.vstack((canvas_attribute_stack, one_week_attr))

    print("CANVAS ATTR SHAPE")
    print(canvas_attribute_stack.shape)
    np.save("canvas_attribute_stack.npy", canvas_attribute_stack)

    #attributes = np.load(os.path.join(dir_path,'sbe_att.npy'))
    attributes = np.load('C:\\Users\\phili\\Desktop\\sbe_att.npy')

    train_sample_size = attributes.shape[0]

    #features = np.load(os.path.join(dir_path,'sbe_feat.npy'))
    features = np.load('C:\\Users\\phili\\Desktop\\sbe_feat.npy')
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
    sample_len = 17

    # normalise data
    (data_feature, data_attribute, data_attribute_outputs, real_attribute_mask) = normalize_per_sample(features, attributes, data_feature_outputs, data_attribute_outputs)

    ### Stick attributes loaded from the file and those loaded from the canvas together for normalization of the canvas attributes ###########################
    print("DATA ATTR SHAPE")
    print(data_attribute.shape)
    comb_attribute = np.vstack((canvas_attribute_stack, data_attribute))

    # add generation flag to features
    data_feature, data_feature_outputs = add_gen_flag(
        data_feature, gen_flags, data_feature_outputs, sample_len)

    data_attribute_min = np.amin(comb_attribute, axis=0)
    data_attribute_max = np.amax(comb_attribute, axis=0)

    norm_attribute = normalize_attribute(comb_attribute, data_attribute_outputs, data_attribute_min, data_attribute_max)

    norm_canvas_attribute = norm_attribute[0:52,:] #cut the canvas attributes off again

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
            data_gen_flag=gen_flags,
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

    features_gt = np.load('C:\\Users\\phili\\Desktop\\europe_feat_test.npy')
    #features_gt = np.load('seasia_feat_test.npy')
    print("Feature loaded")
    features_gt = features_gt.reshape(-1,119,1)
    
    site_list = ['geneva','milan','paris','perlin','zurich']
    #site_list =['hochiminh','jakarta','kualalumpur','pangkok','singapore']

    site_i = 4
    site = site_list[site_i]
    #site_i_gt, site_i_gen = all_annual_batch(features_gt,norm_attribute,1,num_gen,site_i,data_attribute_max[48], gan, sample_len, data_feature_outputs, data_attribute_outputs)
    site_i_gt, site_i_gen = all_annual_batch(features_gt,norm_attribute,1,num_gen,site_i,data_attribute_max[48], gan, sample_len, data_feature_outputs, data_attribute_outputs, norm_canvas_attribute)
    site_i_gt_file = site+'_gt_feat_test.npy'
    site_i_gen_file = site+'_gen_feat_test.npy' 
    
    np.save(site_i_gt_file,site_i_gt)
    np.save(site_i_gen_file,site_i_gen)

    #TEMP TODO remove
    zurich = np.load("zurich_gen_feat_test.npy")
    zurich_mean = np.mean(zurich, 2)

    np.savetxt("zurich_gen_mean_"+str(num_gen)+".txt", zurich_mean)

    zurich_list = zurich_mean[0,:].tolist()

    previous_result = zurich_list

    return zurich_list, DNI_weekly_avg_list, DNI_weekly_peak_list, DHI_weekly_avg_list, DHI_weekly_peak_list, DNI_max_avg_list, DNI_min_avg_list, DHI_max_avg_list, DHI_min_avg_list

if __name__ == "__main__":
    app.run(debug=True)