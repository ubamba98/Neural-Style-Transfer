import numpy as np
class CONFIG:
    IMAGE_WIDTH = 400
    IMAGE_HEIGHT = 300
    COLOR_CHANNELS = 3
    NOISE_RATIO = 0.2
    MEANS = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3)) 
    VGG_MODEL = 'pretrained-model/imagenet-vgg-verydeep-19.mat' # Pick the VGG 19-layer model by from the paper "Very Deep Convolutional Networks for Large-Scale Image Recognition".
    VGG_MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat'
    STYLE_IMAGE = '' # Style image to use.
    CONTENT_IMAGE = '' # Content image to use.
    OUTPUT_DIR = 'output/'
    STYLE_LAYERS = [('conv1_1', 0.2),
                    ('conv2_1', 0.2),
                    ('conv3_1', 0.2),
                    ('conv4_1', 0.2),
                    ('conv5_1', 0.2)]
    LEARNINF_RATE = 2.0
    NUM_ITERATIONS = 200
    ALPHA = 10
    BETA = 40
