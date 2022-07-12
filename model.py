#####################################
# LAN RAW-to-RGB Model architecture #
#####################################

import tensorflow as tf
import numpy as np

def lan_g(input_image):
    activation='lrelu'
    end_activation='tanh'
    with tf.compat.v1.variable_scope("generator"):
        x_a = _conv_layer(input_image, 16, 4, 2, activation=activation) # flat-> layers
        x_b = _downscale(x_a, 16, 3, 2, 'stride', norm='none', sn=False, activation=activation) # downscale
        x_c = _conv_layer(x_b, 16, 3, 1, activation=activation) # first layer
        
        dam1 = _double_att(x_c, activation, mid_activation='none', end_activation=end_activation, reduction=4, multiplier=2)
        dam1 = x_c + dam1

        dam2 = _double_att(dam1, activation, mid_activation='none', end_activation=end_activation, reduction=4, multiplier=2)
        dam2 = dam1 + dam2

        y = _conv_layer(dam2, 16, 3, 1, activation=activation)
        y = x_c + y

        z = _upscale(y, 16, 3, 2, 'transpose', activation='none')
        z = _stack(x_a, z)
        z = _conv_layer(z, 16, 3, 1, activation=activation)
        z = _upscale(z, 64, 3, 2, 'd2s', activation=activation)
        z = _conv_layer(z, 3, 3, 1, activation=activation)
        out = _switch_activation(z, activation=end_activation)
    return out

def _switch_activation(x, activation='none'):
    if activation == 'relu':
        return tf.nn.relu(x)
    elif activation == 'lrelu':
        return tf.nn.leaky_relu(x)
    elif activation == 'gelu':
        return _gelu(x)
    elif activation == 'tanh':
        return tf.nn.tanh(x) * 0.58 + 0.5
    elif activation == 'sigmoid':
        return tf.nn.sigmoid(x)
    elif activation == 'none':
        return x
    else:
        print("Activation not recognized, using none")
        return x

def _gelu(x):
    return 0.5 * x * (1 + tf.tanh(tf.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))

def _upscale(net, num_filters, filter_size, factor, method, activation='lrelu'):
    if method == "transpose":
        return _conv_tranpose_layer(net, num_filters, filter_size, factor, activation)
    elif method == "d2s":
        return tf.nn.depth_to_space(net, 2)
    else:
        print("Unrecognized upscaling method, using transpose")
        return _conv_tranpose_layer(net, num_filters, filter_size, factor, activation)


def _downscale(net, num_filters, filter_size, factor, method, norm, sn, activation='lrelu', padding='SAME'):
    if method == "stride":
        return _conv_layer(net, num_filters, filter_size, factor, norm=norm, padding=padding, activation=activation)
    else:
        print("Unrecognized downscaling method")

def weight_variable(shape, name):
    initial = tf.compat.v1.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial, name=name)

def _stack(x, y):
    return tf.concat([x, y], 3)

def bias_variable(shape, name):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial, name=name)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def _conv_layer(net, num_filters, filter_size, strides, norm='none', padding='SAME', activation='none', use_bias=True, sn=False, actfirst=False):
    weights_init = _conv_init_vars(net, num_filters, filter_size)
    strides_shape = [1, strides, strides, 1]

    if sn:
        weights_init = _spectral_norm(weights_init)

    net = tf.nn.conv2d(net, weights_init, strides_shape, padding=padding)

    if use_bias:
        bias = tf.Variable(tf.constant(0.01, shape=[num_filters]))
        net = tf.nn.bias_add(net, bias)

    if actfirst:
        net = _switch_activation(net, activation)
        net = _switch_norm(net, norm)
    else:
        net = _switch_norm(net, norm)
        net = _switch_activation(net, activation) 
    return net

def _switch_norm(net, norm):
    if norm == 'instance':
        return _instance_norm(net)
    elif norm == 'group':
        return _group_norm(net)
    elif norm == 'layer':
        return _layer_norm(net)
    elif norm == 'none':
        return net
    else:
        print("Norm not recognized, using none")
        return net

def _instance_norm(net):
    batch, rows, cols, channels = [i for i in net.get_shape()]
    var_shape = [channels]

    mu, sigma_sq = tf.compat.v1.nn.moments(net, [1,2], keepdims=True)
    shift = tf.Variable(tf.zeros(var_shape))
    scale = tf.Variable(tf.ones(var_shape))

    epsilon = 1e-3
    normalized = (net-mu)/(sigma_sq + epsilon)**(.5)

    return scale * normalized + shift

def _group_norm(x, G=32, eps=1e-5) :
    N, H, W, C = [i for i in x.get_shape()]
    G = min(G, C)

    x = tf.reshape(x, [N, H, W, G, C // G])
    mean, var = tf.compat.v1.nn.moments(x, [1, 2, 4], keep_dims=True)
    x = (x - mean) / tf.sqrt(var + eps)

    gamma = tf.Variable(tf.constant(1.0, shape = [1, 1, 1, C]))
    beta = tf.Variable(tf.constant(0.0, shape = [1, 1, 1, C]))

    x = tf.reshape(x, [N, H, W, C]) * gamma + beta

    return x

def _layer_norm(net):
    if len(net.get_shape()) == 4:
        batch, rows, cols, channels = [i for i in net.get_shape()]
        axes = [1,2,3]
    elif len(net.get_shape()) == 3:
        batch, vals, channels = [i for i in net.get_shape()]
        axes = [1,2]
    var_shape = [1,1,1,1]

    mu, sigma_sq = tf.compat.v1.nn.moments(net, axes, keepdims=True)
    shift = tf.Variable(tf.zeros(var_shape))
    scale = tf.Variable(tf.ones(var_shape))

    epsilon = 1e-3
    normalized = (net-mu)/(sigma_sq + epsilon)**(.5)

    return scale * normalized + shift

def _conv_init_vars(net, out_channels, filter_size, transpose=False):

    _, rows, cols, in_channels = [i for i in net.get_shape()]

    if not transpose:
        weights_shape = [filter_size, filter_size, in_channels, out_channels]
    else:
        weights_shape = [filter_size, filter_size, out_channels, in_channels]

    weights_init = tf.Variable(tf.compat.v1.truncated_normal(weights_shape, stddev=0.01, seed=1), dtype=tf.float32)
    return weights_init


def _conv_tranpose_layer(net, num_filters, filter_size, strides, activation='lrelu', use_bias=True, sn=False):
    weights_init = _conv_init_vars(net, num_filters, filter_size, transpose=True)

    batch_size, rows, cols, in_channels = [i for i in net.get_shape()]
    new_rows, new_cols = int(rows * strides), int(cols * strides)

    new_shape = [batch_size, new_rows, new_cols, num_filters]
    tf_shape = tf.stack(new_shape)

    strides_shape = [1, strides, strides, 1]

    if sn:
        weights_init = _spectral_norm(weights_init)

    net = tf.nn.conv2d_transpose(net, weights_init, tf_shape, strides_shape, padding='SAME')

    if use_bias:
        bias = tf.Variable(tf.constant(0.01, shape=[num_filters]))
        net = tf.nn.bias_add(net, bias)
    
    net = _switch_activation(net, activation)
    
    return net

def _spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.Variable(tf.random.normal([1, w_shape[-1]]), dtype=tf.float32)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = tf.nn.l2_normalize(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_)

    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = w / sigma
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm

def _double_att(input, activation='relu', mid_activation='none', end_activation='sigmoid', norm='none', reduction=1, multiplier=1):
    batch, rows, cols, channels = [i for i in input.get_shape()]

    x = _conv_layer(input, channels*multiplier, 3, 1, activation=activation, norm=norm)
    x = _conv_layer(x, channels*multiplier, 1, 1, activation=mid_activation, norm=norm)

    ca  = _channel_att(x, activation, end_activation, reduction)
    sa = _spatial_att(x, end_activation)

    x = _stack(ca, sa)
    x = _conv_layer(x, channels, 3, 1, activation=mid_activation, norm=norm)
    return x

def _spatial_att(input, end_activation='sigmoid'):
    batch, rows, cols, channels = [i for i in input.get_shape()]

    weights_init = _conv_init_vars(input, 1, 5)
    strides_shape = [1, 1, 1, 1]
    dilation_rate = [2, 2]
    x = tf.nn.depthwise_conv2d(input, weights_init, strides_shape, padding='SAME', dilations=dilation_rate)
    x = _switch_activation(x, end_activation)
    return tf.math.multiply(x, input)

def _channel_att(input, activation='relu', end_activation='sigmoid', reduction=1):
    batch, rows, cols, channels = [i for i in input.get_shape()]
    x = tf.nn.avg_pool(input, ksize=[1, rows, cols, 1], strides=[1,1,1,1], padding='VALID')
    x = _conv_layer(x, channels//reduction, 1, 1, activation=activation)
    x = _conv_layer(x, channels, 1, 1, activation=end_activation)
    return x * input