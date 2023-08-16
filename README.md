# SolarGAN
 Generating synthetic urban solar irradiance time-series via Deep Generative Networks

Data can be shared upon request.

(16.03.2023) 3 notebooks with some necessary utility functions uploaded:

fisheye_image_processing: Transform fisheye renderings into grayscale cubemap images as input for the image encorder.

att_processing: load the image encoder to compress cubemap images into a vector (dim = 32), draw weather statistics & sensor-point location information as another vector (Table 2), and concat. both as input attribute vector.

gen_time_series: load the time series generator and draw time series samples according to the input attribute vector.  

# SolarGAN Setup with Hops

Hops is a Grasshopper package that lets you add external functions to Grasshopper. Specifically, it makes is possible to run "external" Python code from the Grasshopper canvas. We can use Hops to run the python code for loading and executing SolarGANs trained models, making use of SolarGANs capabilities in Grasshopper.

All necessary files to run SolarGAN in Grasshopper through Hops can be found in the hops_test subdirectory.

## Requirements

- Rhino 7.4 or newer.
- CPython 3.8 or above
- Hops Component for Grasshopper

## Required Python packages

- Flask
- ghhops_server

- torch
- cv2
- numpy
- PIL
- tqdm

## Setup steps

1. Install the required Software and Python packages
2. Navigate into the hops_test directory and run 'python ./app.py' from the command line to launch the Hops server
3. Open 'SolarGAN_Hops_Template.gh' in Grasshopper
4. Done!

If the Hops components show an error, try right-clicking and selecting the Path option. The API endpoint should read something like 'http://127.0.0.1:5000/att_processing'. Hops tries to to reach the Hops server at this address. If this does not work, make sure the Hops server is running correctly and that the local address matches the one declared in the Path in Grasshopper.