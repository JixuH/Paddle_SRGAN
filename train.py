import cv2
import os
import tqdm
import time
import shutil
import random
import numpy as np
import paddle.fluid as fluid

from load import *
from network import *

## Adam
batch_size = 32
lr = 0.001
beta1 = 0.9
use_gpu = True

## initialize G
n_epoch_init = 50

## adversarial learning (SRGAN)
n_epoch = 2000

## train set location
train_hr_img_path = '/home/aistudio/srdata/DIV2K_train_HR' 
train_lr_img_path = '/home/aistudio/srdata/DIV2K_train_LR_bicubic/X4'

## test set location
valid_hr_img_path = '/home/aistudio/srdata/DIV2K_valid_HR'
valid_lr_img_path = '/home/aistudio/srdata/DIV2K_valid_LR_bicubic/X4'

# load im path to list
train_hr_img_list = sorted(load_file_list(im_path=train_hr_img_path, im_format='*.png'))
train_lr_img_list = sorted(load_file_list(im_path=train_lr_img_path, im_format='*.png'))
valid_hr_img_list = sorted(load_file_list(im_path=valid_hr_img_path, im_format='*.png'))
valid_lr_img_list = sorted(load_file_list(im_path=valid_lr_img_path, im_format='*.png'))

# load im data
train_hr_imgs = im_read(train_hr_img_list)
train_lr_imgs = im_read(train_lr_img_list)
valid_hr_imgs = im_read(valid_hr_img_list)
valid_lr_imgs = im_read(valid_lr_img_list)

# LOAD VGG
vgg19_program =fluid.Program()
with fluid.program_guard(vgg19_program): 
    vgg19_input = fluid.layers.data(name='vgg19_input',shape=[224, 224, 3],dtype='float32')
    vgg19_input_transpose = fluid.layers.transpose(vgg19_input, perm=[0, 3, 1, 2])
    # define vgg19
    _, vgg_target_emb = vgg19(vgg19_input_transpose)


# DEFINE MODEL ==> SRGAN_g SRGAN_d
SRGAN_g_program =fluid.Program()
with fluid.program_guard(SRGAN_g_program):
    # Low resolution image
    t_image = fluid.layers.data(name='t_image',shape=[96, 96, 3],dtype='float32')
    #print(t_image.shape)
    t_image_transpose = fluid.layers.transpose(t_image, perm=[0, 3, 1, 2])
    #print(t_image_transpose.shape)
    # High resolution image
    t_target_image = fluid.layers.data(name='t_target_image',shape=[384, 384, 3],dtype='float32')
    t_target_image_transpose = fluid.layers.transpose(t_target_image, perm=[0, 3, 1, 2])
    # define SRGAN_g
    net_g = SRGAN_g(t_image_transpose)
    #net_g_test = SRGAN_g(t_image_transpose)
    test_im = fluid.layers.transpose(net_g, perm=[0, 2, 3, 1])
    # vgg19_input
    vgg19_input = fluid.layers.data(name='vgg19_input',shape=[224, 224, 3],dtype='float32')
    vgg19_input_transpose = fluid.layers.transpose(vgg19_input, perm=[0, 3, 1, 2])
    # get vgg_target_emb vgg_predict_emb
    t_predict_image_224 = fluid.layers.image_resize(input=net_g, out_shape=[224, 224], resample="NEAREST")
    _, vgg_target_emb = vgg19(vgg19_input_transpose)
    _, vgg_predict_emb = vgg19(t_predict_image_224)
    # get logits_fake
    logits_fake = SRGAN_d(net_g)
    # g_loss mse_loss
    g_loss, mse_loss = calc_g_loss(net_g, t_target_image_transpose, logits_fake, vgg_predict_emb, vgg_target_emb)

SRGAN_d_program =fluid.Program()
with fluid.program_guard(SRGAN_d_program):
    # Low resolution image
    t_image = fluid.layers.data(name='t_image',shape=[96, 96, 3],dtype='float32')
    t_image_transpose = fluid.layers.transpose(t_image, perm=[0, 3, 1, 2])
    # High resolution image
    t_target_image = fluid.layers.data(name='t_target_image',shape=[384, 384, 3],dtype='float32')
    t_target_image_transpose = fluid.layers.transpose(t_target_image, perm=[0, 3, 1, 2])
    net_g = SRGAN_g(t_image_transpose)
    # define SRGAN_d
    logits_real = SRGAN_d(t_target_image_transpose)
    logits_fake = SRGAN_d(net_g)
    d_loss = calc_d_loss(logits_real, logits_fake)

place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
exe = fluid.Executor(place)
# 进行参数初始化
exe.run(fluid.default_startup_program())

def load_vars(exe, program, pretrained_model):
    vars = []
    for var in program.list_vars():
        if fluid.io.is_parameter(var) and var.name.startswith("vgg"):
            vars.append(var)
            print(var.name)
    fluid.io.load_vars(exe, pretrained_model, program, vars)

save_pretrain_model_path = './VGG19_pretrained'
load_vars(exe, vgg19_program, save_pretrain_model_path)

# init
t_image = fluid.layers.data(name='t_image',shape=[96, 96, 3],dtype='float32')
t_target_image = fluid.layers.data(name='t_target_image',shape=[384, 384, 3],dtype='float32')
vgg19_input = fluid.layers.data(name='vgg19_input',shape=[224, 224, 3],dtype='float32')

step_num = int(len(train_hr_imgs) / batch_size)
# initialize G
for epoch in range(0, n_epoch_init + 1):
    epoch_time = time.time()
    np.random.shuffle(train_hr_imgs)
    # real
    sample_imgs_384 = random_crop(train_hr_imgs, 384)
    sample_imgs_standardized_384 = standardized(sample_imgs_384)
    # input
    sample_imgs_96 = im_resize(sample_imgs_384,96,96)
    sample_imgs_standardized_96 = standardized(sample_imgs_96)
    # vgg19
    sample_imgs_224 = im_resize(sample_imgs_384,224,224)
    sample_imgs_standardized_224 = standardized(sample_imgs_224)
    # loss
    total_mse_loss, n_iter = 0, 0
    for i in tqdm.tqdm(range(step_num)):
        step_time = time.time()
        imgs_384 = sample_imgs_standardized_384[i * batch_size:(i + 1) * batch_size]
        imgs_384 = np.array(imgs_384, dtype='float32')
        imgs_96 = sample_imgs_standardized_96[i * batch_size:(i + 1) * batch_size]
        imgs_96 = np.array(imgs_96, dtype='float32')
        # vgg19 data
        imgs_224 = sample_imgs_standardized_224[i * batch_size:(i + 1) * batch_size]
        imgs_224 = np.array(imgs_224, dtype='float32')
        # print(imgs_384.shape)
        # print(imgs_96.shape)
        # print(imgs_224.shape)
        # update G
        mse_loss_n = exe.run(SRGAN_g_program,
                        feed={'t_image': imgs_96, 't_target_image': imgs_384, 'vgg19_input':imgs_224},
                        fetch_list=[mse_loss])[0]
        #print(mse_loss_n)
        #print("Epoch [%2d/%2d] %4d time: %4.4fs, mse: %.8f " % (epoch, n_epoch_init, n_iter, time.time() - step_time, mse_loss_n))
        total_mse_loss += mse_loss_n
        n_iter += 1
    log = "[*] Epoch_init: [%2d/%2d] time: %4.4fs, mse: %.8f" % (epoch, n_epoch_init, time.time() - epoch_time, total_mse_loss / n_iter)
    print(log)

    if (epoch != 0) and (epoch % 10 == 0):
        out = exe.run(SRGAN_g_program,
                        feed={'t_image': imgs_96, 't_target_image': imgs_384, 'vgg19_input':imgs_224},
                        fetch_list=[test_im])[0][0]
        # generate img
        im_G = np.array((out+1)*127.5, dtype=np.uint8)
        im_96 = np.array((imgs_96[0]+1)*127.5, dtype=np.uint8)
        im_384 = np.array((imgs_384[0]+1)*127.5, dtype=np.uint8)
        cv2.imwrite('./output/epoch_init_{}_G.jpg'.format(epoch), cv2.cvtColor(im_G, cv2.COLOR_RGB2BGR))
        cv2.imwrite('./output/epoch_init_{}_96.jpg'.format(epoch), cv2.cvtColor(im_96, cv2.COLOR_RGB2BGR))
        cv2.imwrite('./output/epoch_init_{}_384.jpg'.format(epoch), cv2.cvtColor(im_384, cv2.COLOR_RGB2BGR))
    
    # # save model
    # save_pretrain_model_path_init = 'models/init/'
    # # delete old model files
    # shutil.rmtree(save_pretrain_model_path_init, ignore_errors=True)
    # # mkdir
    # os.makedirs(save_pretrain_model_path_init)
    # fluid.io.save_persistables(executor=exe, dirname=save_pretrain_model_path_init, main_program=SRGAN_g_program)

# train GAN (SRGAN)
for epoch in range(0, n_epoch + 1):
    ## update learning rate
    epoch_time = time.time()
    
    # real
    sample_imgs_384 = random_crop(train_hr_imgs, 384)
    sample_imgs_standardized_384 = standardized(sample_imgs_384)
    # input
    sample_imgs_96 = im_resize(sample_imgs_384,96,96)
    sample_imgs_standardized_96 = standardized(sample_imgs_96)
    # vgg19
    sample_imgs_224 = im_resize(sample_imgs_384,224,224)
    sample_imgs_standardized_224 = standardized(sample_imgs_224)
    # loss
    total_d_loss, total_g_loss, n_iter = 0, 0, 0
    for i in tqdm.tqdm(range(step_num)):
        step_time = time.time()
        imgs_384 = sample_imgs_standardized_384[i * batch_size:(i + 1) * batch_size]
        imgs_384 = np.array(imgs_384, dtype='float32')
        imgs_96 = sample_imgs_standardized_96[i * batch_size:(i + 1) * batch_size]
        imgs_96 = np.array(imgs_96, dtype='float32')
        # vgg19 data
        imgs_224 = sample_imgs_standardized_224[i * batch_size:(i + 1) * batch_size]
        imgs_224 = np.array(imgs_224, dtype='float32')
        ## update D
        errD = exe.run(SRGAN_d_program,
                        feed={'t_image': imgs_96, 't_target_image': imgs_384},
                        fetch_list=[d_loss])[0]
        ## update G
        errG = exe.run(SRGAN_g_program,
                        feed={'t_image': imgs_96, 't_target_image': imgs_384, 'vgg19_input':imgs_224},
                        fetch_list=[g_loss])[0]
        # print("Epoch [%2d/%2d] %4d time: %4.4fs, d_loss: %.8f g_loss: %.8f (mse: %.6f vgg: %.6f adv: %.6f)" %
        #       (epoch, n_epoch, n_iter, time.time() - step_time, errD, errG, errM, errV, errA))
        total_d_loss += errD
        total_g_loss += errG
        n_iter += 1
    log = "[*] Epoch: [%2d/%2d] time: %4.4fs, d_loss: %.8f g_loss: %.8f" % (epoch, n_epoch, time.time() - epoch_time, total_d_loss / n_iter, total_g_loss / n_iter)
    print(log)

    if (epoch != 0) and (epoch % 10 == 0):
        out = exe.run(SRGAN_g_program,
                        feed={'t_image': imgs_96, 't_target_image': imgs_384, 'vgg19_input':imgs_224},
                        fetch_list=[test_im])[0][0]
        # generate img
        im_G = np.array((out + 1) * 127.5, dtype=np.uint8)
        im_96 = np.array((imgs_96[0] + 1) * 127.5, dtype=np.uint8)
        im_384 = np.array((imgs_384[0] + 1) * 127.5, dtype=np.uint8)
        cv2.imwrite('./output/epoch_{}_G.jpg'.format(epoch), cv2.cvtColor(im_G, cv2.COLOR_RGB2BGR))
        cv2.imwrite('./output/epoch_{}_96.jpg'.format(epoch), cv2.cvtColor(im_96, cv2.COLOR_RGB2BGR))
        cv2.imwrite('./output/epoch_{}_384.jpg'.format(epoch), cv2.cvtColor(im_384, cv2.COLOR_RGB2BGR))
    # save model
    # d_models
    save_pretrain_model_path_d = 'models/d_models/'
    # delete old model files
    shutil.rmtree(save_pretrain_model_path_d, ignore_errors=True)
    # mkdir
    os.makedirs(save_pretrain_model_path_d)
    fluid.io.save_persistables(executor=exe, dirname=save_pretrain_model_path_d, main_program=SRGAN_g_program)
    # g_models
    save_pretrain_model_path_g = 'models/g_models/'
    # delete old model files
    shutil.rmtree(save_pretrain_model_path_g, ignore_errors=True)
    # mkdir
    os.makedirs(save_pretrain_model_path_g)
    fluid.io.save_persistables(executor=exe, dirname=save_pretrain_model_path_g, main_program=SRGAN_g_program)