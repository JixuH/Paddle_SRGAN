import paddle
import paddle.fluid as fluid

# SRGAN_g
def SRGAN_g(t_image):
    # Input-Conv-Relu
    n = fluid.layers.conv2d(input=t_image, num_filters=64, filter_size=3, stride=1, padding='SAME', name='n64s1/c', data_format='NCHW')
    # print('conv0', n.shape)
    n = fluid.layers.batch_norm(n, momentum=0.99, epsilon=0.001)
    n = fluid.layers.relu(n, name=None)
    temp = n

    # B residual blocks
    # Conv-BN-Relu-Conv-BN-Elementwise_add
    for i in range(16):
        nn = fluid.layers.conv2d(input=n, num_filters=64, filter_size=3, stride=1, padding='SAME', name='n64s1/c1/%s' % i, data_format='NCHW')
        nn = fluid.layers.batch_norm(nn, momentum=0.99, epsilon=0.001, name='n64s1/b1/%s' % i)
        nn = fluid.layers.relu(nn, name=None)
        log = 'conv%2d' % (i+1)
        # print(log, nn.shape)
        nn = fluid.layers.conv2d(input=nn, num_filters=64, filter_size=3, stride=1, padding='SAME', name='n64s1/c2/%s' % i, data_format='NCHW')
        nn = fluid.layers.batch_norm(nn, momentum=0.99, epsilon=0.001, name='n64s1/b2/%s' % i)
        nn = fluid.layers.elementwise_add(n, nn, act=None, name='b_residual_add/%s' % i)
        n = nn

    n = fluid.layers.conv2d(input=n, num_filters=64, filter_size=3, stride=1, padding='SAME', name='n64s1/c/m', data_format='NCHW')
    n = fluid.layers.batch_norm(n, momentum=0.99, epsilon=0.001, name='n64s1/b2/%s' % i)
    n = fluid.layers.elementwise_add(n, temp, act=None, name='add3')
    # print('conv17', n.shape)

    # B residual blacks end
    # Conv-Pixel_shuffle-Conv-Pixel_shuffle-Conv
    n = fluid.layers.conv2d(input=n, num_filters=256, filter_size=3, stride=1, padding='SAME', name='n256s1/1', data_format='NCHW')
    n = fluid.layers.pixel_shuffle(n, upscale_factor=2)
    n = fluid.layers.relu(n, name=None)
    # print('conv18', n.shape)

    n = fluid.layers.conv2d(input=n, num_filters=256, filter_size=3, stride=1, padding='SAME', name='n256s1/2', data_format='NCHW')
    n = fluid.layers.pixel_shuffle(n, upscale_factor=2)
    n = fluid.layers.relu(n, name=None)
    # print('conv19', n.shape)
    n = fluid.layers.conv2d(input=n, num_filters=3, filter_size=1, stride=1, padding='SAME', name='out', data_format='NCHW')
    n = fluid.layers.tanh(n, name=None)
    # print('conv20', n.shape)

    return n


## SRGAN_d
def SRGAN_d(input_images):
    # Conv-Leaky_Relu 
    net_h0 = fluid.layers.conv2d(input=input_images, num_filters=64, filter_size=4, stride=2, padding='SAME', name='h0/c', data_format='NCHW')
    net_h0 = fluid.layers.leaky_relu(net_h0, alpha=0.2, name=None)
    # h1 Cnov-BN-Leaky_Relu
    net_h1 = fluid.layers.conv2d(input=net_h0, num_filters=128, filter_size=4, stride=2, padding='SAME', name='h1/c', data_format='NCHW')
    net_h1 = fluid.layers.batch_norm(net_h1, momentum=0.99, epsilon=0.001, name='h1/bn')  
    net_h1 = fluid.layers.leaky_relu(net_h1, alpha=0.2, name=None)
    # h2 Cnov-BN-Leaky_Relu
    net_h2 = fluid.layers.conv2d(input=net_h1, num_filters=256, filter_size=4, stride=2, padding='SAME', name='h2/c', data_format='NCHW')
    net_h2 = fluid.layers.batch_norm(net_h2, momentum=0.99, epsilon=0.001, name='h2/bn')
    net_h2 = fluid.layers.leaky_relu(net_h2, alpha=0.2, name=None)
    # h3 Cnov-BN-Leaky_Relu
    net_h3 = fluid.layers.conv2d(input=net_h2, num_filters=512, filter_size=4, stride=2, padding='SAME', name='h3/c', data_format='NCHW')
    net_h3 = fluid.layers.batch_norm(net_h3, momentum=0.99, epsilon=0.001, name='h3/bn')
    net_h3 = fluid.layers.leaky_relu(net_h3, alpha=0.2, name=None)
    # h4 Cnov-BN-Leaky_Relu
    net_h4 = fluid.layers.conv2d(input=net_h3, num_filters=1024, filter_size=4, stride=2, padding='SAME', name='h4/c', data_format='NCHW')
    net_h4 = fluid.layers.batch_norm(net_h4, momentum=0.99, epsilon=0.001, name='h4/bn')
    net_h4 = fluid.layers.leaky_relu(net_h4, alpha=0.2, name=None)
    # h5 Cnov-BN-Leaky_Relu
    net_h5 = fluid.layers.conv2d(input=net_h4, num_filters=2048, filter_size=4, stride=2, padding='SAME', name='h5/c', data_format='NCHW')
    net_h5 = fluid.layers.batch_norm(net_h5, momentum=0.99, epsilon=0.001, name='h5/bn')
    net_h5 = fluid.layers.leaky_relu(net_h5, alpha=0.2, name=None)
    # h6 Cnov-BN-Leaky_Relu
    net_h6 = fluid.layers.conv2d(input=net_h5, num_filters=1024, filter_size=4, stride=2, padding='SAME', name='h6/c', data_format='NCHW')
    net_h6 = fluid.layers.batch_norm(net_h6, momentum=0.99, epsilon=0.001, name='h6/bn')
    net_h6 = fluid.layers.leaky_relu(net_h6, alpha=0.2, name=None)
    # h7 Cnov-BN-Leaky_Relu
    net_h7 = fluid.layers.conv2d(input=net_h6, num_filters=512, filter_size=4, stride=2, padding='SAME', name='h7/c', data_format='NCHW')
    net_h7 = fluid.layers.batch_norm(net_h7, momentum=0.99, epsilon=0.001, name='h7/bn')
    net_h7 = fluid.layers.leaky_relu(net_h7, alpha=0.2, name=None)
    #修改原论文网络
    net = fluid.layers.conv2d(input=net_h7, num_filters=128, filter_size=1, stride=1, padding='SAME', name='res/c', data_format='NCHW')
    net = fluid.layers.batch_norm(net, momentum=0.99, epsilon=0.001, name='res/bn')
    net = fluid.layers.leaky_relu(net, alpha=0.2, name=None)
    net = fluid.layers.conv2d(input=net_h7, num_filters=128, filter_size=3, stride=1, padding='SAME', name='res/c2', data_format='NCHW')
    net = fluid.layers.batch_norm(net, momentum=0.99, epsilon=0.001, name='res/bn2')
    net = fluid.layers.leaky_relu(net, alpha=0.2, name=None)
    net = fluid.layers.conv2d(input=net_h7, num_filters=512, filter_size=3, stride=1, padding='SAME', name='res/c3', data_format='NCHW')
    net = fluid.layers.batch_norm(net, momentum=0.99, epsilon=0.001, name='res/bn3')
    net = fluid.layers.leaky_relu(net, alpha=0.2, name=None)

    net_h8 = fluid.layers.elementwise_add(net_h7, net, act=None, name='res/add')
    net_h8 = fluid.layers.leaky_relu(net_h8, alpha=0.2, name=None)

    #net_ho = fluid.layers.flatten(net_h8, axis=0, name='ho/flatten')
    net_ho = fluid.layers.fc(input=net_h8, size=1024, name='ho/fc')
    net_ho = fluid.layers.leaky_relu(net_ho, alpha=0.2, name=None)
    net_ho = fluid.layers.fc(input=net_h8, size=1, name='ho/fc2')
    # return
    # logits = net_ho
    net_ho = fluid.layers.sigmoid(net_ho, name=None)

    return net_ho # , logits


## vgg19
def conv_block(input, num_filter, groups, name=None):
    conv = input
    for i in range(groups):
        conv = fluid.layers.conv2d(
            input=conv,
            num_filters=num_filter,
            filter_size=3,
            stride=1,
            padding=1,
            act='relu',
            param_attr=fluid.param_attr.ParamAttr(
                name=name + str(i + 1) + "_weights"),
            bias_attr=False)
    return fluid.layers.pool2d(
        input=conv, pool_size=2, pool_type='max', pool_stride=2)

def vgg19(input, class_dim=1000):

    # VGG_MEAN = [123.68, 103.939, 116.779]

    # """ input layer """
    # net_in = (input + 1) * 127.5

    # red, green, blue = fluid.layers.split(net_in, num_or_sections=3, dim=1)

    # net_in = fluid.layers.concat(input=[red-VGG_MEAN[0], green-VGG_MEAN[1], blue-VGG_MEAN[2]], axis=0)

    layers = 19
    vgg_spec = {
        11: ([1, 1, 2, 2, 2]),
        13: ([2, 2, 2, 2, 2]),
        16: ([2, 2, 3, 3, 3]),
        19: ([2, 2, 4, 4, 4])
    }
    assert layers in vgg_spec.keys(), \
        "supported layers are {} but input layer is {}".format(vgg_spec.keys(), layers)

    nums = vgg_spec[layers]
    conv1 = conv_block(input, 64, nums[0], name="vgg19_conv1_")
    conv2 = conv_block(conv1, 128, nums[1], name="vgg19_conv2_")
    conv3 = conv_block(conv2, 256, nums[2], name="vgg19_conv3_")
    conv4 = conv_block(conv3, 512, nums[3], name="vgg19_conv4_")
    conv5 = conv_block(conv4, 512, nums[4], name="vgg19_conv5_")

    fc_dim = 4096
    fc_name = ["fc6", "fc7", "fc8"]
    fc1 = fluid.layers.fc(
        input=conv5,
        size=fc_dim,
        act='relu',
        param_attr=fluid.param_attr.ParamAttr(
            name=fc_name[0] + "_weights"),
        bias_attr=fluid.param_attr.ParamAttr(name=fc_name[0] + "_offset"))
    fc1 = fluid.layers.dropout(x=fc1, dropout_prob=0.5)
    fc2 = fluid.layers.fc(
        input=fc1,
        size=fc_dim,
        act='relu',
        param_attr=fluid.param_attr.ParamAttr(
            name=fc_name[1] + "_weights"),
        bias_attr=fluid.param_attr.ParamAttr(name=fc_name[1] + "_offset"))
    fc2 = fluid.layers.dropout(x=fc2, dropout_prob=0.5)
    out = fluid.layers.fc(
        input=fc2,
        size=class_dim,
        param_attr=fluid.param_attr.ParamAttr(
            name=fc_name[2] + "_weights"),
        bias_attr=fluid.param_attr.ParamAttr(name=fc_name[2] + "_offset"))

    return out, conv5

# calc_g_loss
def calc_g_loss(net_g, t_target_image, logits_fake, vgg_predict_emb, vgg_target_emb):
    g_gan_loss = fluid.layers.reduce_mean(1e-3 * fluid.layers.sigmoid_cross_entropy_with_logits(x=logits_fake, label=fluid.layers.zeros_like(logits_fake)))
    # g_gan_loss = 1e-3 * fluid.layers.sigmoid_cross_entropy_with_logits(x=logits_fake, label=fluid.layers.zeros_like(logits_fake))
    mse_loss = fluid.layers.reduce_mean(fluid.layers.square_error_cost(net_g, t_target_image))
    vgg_loss = fluid.layers.reduce_mean(2e-6 * fluid.layers.square_error_cost(vgg_predict_emb, vgg_target_emb))
    g_loss = fluid.layers.reduce_mean(g_gan_loss + mse_loss + vgg_loss)
    return g_loss, mse_loss
# calc_d_loss
def calc_d_loss(logits_real, logits_fake):
    d_loss_real = fluid.layers.reduce_mean(fluid.layers.sigmoid_cross_entropy_with_logits(x=logits_real, label=fluid.layers.ones_like(logits_real)))
    d_loss_fake = fluid.layers.reduce_mean(fluid.layers.sigmoid_cross_entropy_with_logits(x=logits_fake, label=fluid.layers.zeros_like(logits_fake)))
    d_loss = fluid.layers.elementwise_add(d_loss_real, d_loss_fake)/2
    return d_loss