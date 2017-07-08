__author__ = 'Fabian Isensee'
from collections import OrderedDict
from lasagne.layers import (InputLayer, ConcatLayer, Pool2DLayer, ReshapeLayer, DimshuffleLayer, NonlinearityLayer,
                            DropoutLayer, Deconv2DLayer, batch_norm, DenseLayer)
try:
    from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
except ImportError:
    from lasagne.layers import Conv2DLayer as ConvLayer
import lasagne
from lasagne.init import HeNormal
from lasagne.nonlinearities import LeakyRectify, sigmoid
lref = LeakyRectify(0.2)


def build_UNet(inputVar=None, nonlinearity=lasagne.nonlinearities.elu, input_dim=(128, 128), base_n_filters=64, do_dropout=False):
    net = OrderedDict()
    pad = "same"
    if not inputVar:
        net['input'] = InputLayer((None, 3, input_dim[0], input_dim[1]))
    else:
        net['input'] = InputLayer((None, 3, input_dim[0], input_dim[1]), inputVar)
    
    net['contr_1_1'] = batch_norm(ConvLayer(net['input'], base_n_filters, 3, nonlinearity=nonlinearity, pad=pad, W=HeNormal(gain="relu")))
    net['contr_1_2'] = batch_norm(ConvLayer(net['contr_1_1'], base_n_filters, 3, nonlinearity=nonlinearity, pad=pad, W=HeNormal(gain="relu")))
    net['pool1'] = Pool2DLayer(net['contr_1_2'], 2)

    net['contr_2_1'] = batch_norm(ConvLayer(net['pool1'], base_n_filters*2, 3, nonlinearity=nonlinearity, pad=pad, W=HeNormal(gain="relu")))
    net['contr_2_2'] = batch_norm(ConvLayer(net['contr_2_1'], base_n_filters*2, 3, nonlinearity=nonlinearity, pad=pad, W=HeNormal(gain="relu")))
    net['pool2'] = Pool2DLayer(net['contr_2_2'], 2)

    net['contr_3_1'] = batch_norm(ConvLayer(net['pool2'], base_n_filters*4, 3, nonlinearity=nonlinearity, pad=pad, W=HeNormal(gain="relu")))
    net['contr_3_2'] = batch_norm(ConvLayer(net['contr_3_1'], base_n_filters*4, 3, nonlinearity=nonlinearity, pad=pad, W=HeNormal(gain="relu")))
    net['pool3'] = Pool2DLayer(net['contr_3_2'], 2)

    net['contr_4_1'] = batch_norm(ConvLayer(net['pool3'], base_n_filters*8, 3, nonlinearity=nonlinearity, pad=pad, W=HeNormal(gain="relu")))
    net['contr_4_2'] = batch_norm(ConvLayer(net['contr_4_1'], base_n_filters*8, 3, nonlinearity=nonlinearity, pad=pad, W=HeNormal(gain="relu")))
    l = net['pool4'] = Pool2DLayer(net['contr_4_2'], 2)
    # the paper does not really describe where and how dropout is added. Feel free to try more options
    if do_dropout:
        l = DropoutLayer(l, p=0.4)

    net['encode_1'] = batch_norm(ConvLayer(l, base_n_filters*16, 3, nonlinearity=nonlinearity, pad=pad, W=HeNormal(gain="relu")))
    net['encode_2'] = batch_norm(ConvLayer(net['encode_1'], base_n_filters*16, 3, nonlinearity=nonlinearity, pad=pad, W=HeNormal(gain="relu")))
    net['upscale1'] = batch_norm(Deconv2DLayer(net['encode_2'], base_n_filters*16, 2, 2, crop="valid", nonlinearity=nonlinearity, W=HeNormal(gain="relu")))

    net['concat1'] = ConcatLayer([net['upscale1'], net['contr_4_2']], cropping=(None, None, "center", "center"))
    net['expand_1_1'] = batch_norm(ConvLayer(net['concat1'], base_n_filters*8, 3, nonlinearity=nonlinearity, pad=pad, W=HeNormal(gain="relu")))
    net['expand_1_2'] = batch_norm(ConvLayer(net['expand_1_1'], base_n_filters*8, 3, nonlinearity=nonlinearity, pad=pad, W=HeNormal(gain="relu")))
    net['upscale2'] = batch_norm(Deconv2DLayer(net['expand_1_2'], base_n_filters*8, 2, 2, crop="valid", nonlinearity=nonlinearity, W=HeNormal(gain="relu")))

    net['concat2'] = ConcatLayer([net['upscale2'], net['contr_3_2']], cropping=(None, None, "center", "center"))
    net['expand_2_1'] = batch_norm(ConvLayer(net['concat2'], base_n_filters*4, 3, nonlinearity=nonlinearity, pad=pad, W=HeNormal(gain="relu")))
    net['expand_2_2'] = batch_norm(ConvLayer(net['expand_2_1'], base_n_filters*4, 3, nonlinearity=nonlinearity, pad=pad, W=HeNormal(gain="relu")))
    net['upscale3'] = batch_norm(Deconv2DLayer(net['expand_2_2'], base_n_filters*4, 2, 2, crop="valid", nonlinearity=nonlinearity, W=HeNormal(gain="relu")))

    net['concat3'] = ConcatLayer([net['upscale3'], net['contr_2_2']], cropping=(None, None, "center", "center"))
    net['expand_3_1'] = batch_norm(ConvLayer(net['concat3'], base_n_filters*2, 3, nonlinearity=nonlinearity, pad=pad, W=HeNormal(gain="relu")))
    net['expand_3_2'] = batch_norm(ConvLayer(net['expand_3_1'], base_n_filters*2, 3, nonlinearity=nonlinearity, pad=pad, W=HeNormal(gain="relu")))
    net['upscale4'] = batch_norm(Deconv2DLayer(net['expand_3_2'], base_n_filters*2, 2, 2, crop="valid", nonlinearity=nonlinearity, W=HeNormal(gain="relu")))

    net['concat4'] = ConcatLayer([net['upscale4'], net['contr_1_2']], cropping=(None, None, "center", "center"))
    net['expand_4_1'] = batch_norm(ConvLayer(net['concat4'], base_n_filters, 3, nonlinearity=nonlinearity, pad=pad, W=HeNormal(gain="relu")))
    net['expand_4_2'] = batch_norm(ConvLayer(net['expand_4_1'], base_n_filters, 3, nonlinearity=nonlinearity, pad=pad, W=HeNormal(gain="relu")))

    net['output'] = ConvLayer(net['expand_4_2'], 3, 1, nonlinearity=None)

    return net



def build_Discriminator(input_var=None, img_size=[128, 128], base_units=64):
    discr = {}
    
    if not input_var:
        discr["input"] = InputLayer((None, 3, img_size[0], img_size[1]))
    else:
        discr["input"] = InputLayer((None, 3, img_size[0], img_size[1]), input_var)
        
    discr["conv_1"] = ConvLayer(discr["input"], base_units, 5, nonlinearity=lref)
    discr["pool_1"] = Pool2DLayer(discr["input"], 2)

    discr["conv_2"] = ConvLayer(discr["pool_1"], base_units*2, 5, nonlinearity=lref)
    discr["pool_2"] = Pool2DLayer(discr["conv_2"], 2)

    discr["dense_1"] = DenseLayer(discr["pool_2"], 1024, nonlinearity=lref)
    discr["output"] = DenseLayer(discr["dense_1"], 1, nonlinearity=sigmoid)
    
    return discr