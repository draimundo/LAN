#####################
# Utility functions #
#####################

from functools import reduce
import tensorflow as tf
import scipy.stats as st
import numpy as np
import sys
import os

def process_command_args(arguments):
    # Specifying the default parameters for training/validation
    # --- data path ---
    dataset_dir = 'raw_images/'
    vgg_dir = 'vgg_pretrained/imagenet-vgg-verydeep-19.mat'
    dslr_dir = 'fujifilm/'
    phone_dir = 'mediatek_raw/'
    model_dir = 'models/'
    
    # --- model weights ---
    restore_iter = None
    # --- input size ---
    patch_w = 256 # default size for MAI dataset
    patch_h = 256 # default size for MAI dataset
    # --- training options ---
    batch_size = 32
    train_size = 5000
    learning_rate = 5e-5
    eval_step = 1000
    num_train_iters = 100000
    
    # --- optimizer options ---
    optimizer='radam'
    default_facs = True
    fac_mse = 0
    fac_l1 = 0
    fac_ssim = 0
    fac_ms_ssim = 0
    fac_uv = 0
    fac_vgg = 0
    fac_lpips = 0
    fac_huber = 0
    fac_charbonnier = 0

    for args in arguments:
        # --- data path ---
        if args.startswith("dataset_dir"):
            dataset_dir = args.split("=")[1]
        if args.startswith("vgg_dir"):
            vgg_dir = args.split("=")[1]
        if args.startswith("dslr_dir"):
            dslr_dir = args.split("=")[1]
        if args.startswith("phone_dir"):
            phone_dir = args.split("=")[1]
        if args.startswith("model_dir"):
            model_dir = args.split("=")[1]

        # --- model weights ---
        if args.startswith("restore_iter"):
            restore_iter = int(args.split("=")[1])

        # --- input size ---
        if args.startswith("patch_w"):
            patch_w = int(args.split("=")[1])
        if args.startswith("patch_h"):
            patch_h = int(args.split("=")[1])

        # --- training options ---
        if args.startswith("batch_size"):
            batch_size = int(args.split("=")[1])
        if args.startswith("train_size"):
            train_size = int(args.split("=")[1])
        if args.startswith("learning_rate"):
            learning_rate = float(args.split("=")[1])
        if args.startswith("eval_step"):
            eval_step = int(args.split("=")[1])
        if args.startswith("num_train_iters"):
            num_train_iters = int(args.split("=")[1])

        # --- more options ---
        if args.startswith("optimizer"):
            optimizer = args.split("=")[1]
        if args.startswith("fac_mse"):
            fac_mse = float(args.split("=")[1])
            default_facs = False
        if args.startswith("fac_l1"):
            fac_l1 = float(args.split("=")[1])
            default_facs = False
        if args.startswith("fac_ssim"):
            fac_ssim = float(args.split("=")[1])
            default_facs = False
        if args.startswith("fac_ms_ssim"):
            fac_ms_ssim = float(args.split("=")[1])
            default_facs = False
        if args.startswith("fac_uv"):
            fac_uv = float(args.split("=")[1])
            default_facs = False
        if args.startswith("fac_vgg"):
            fac_vgg = float(args.split("=")[1])
            default_facs = False
        if args.startswith("fac_lpips"):
            fac_lpips = float(args.split("=")[1])
            default_facs = False
        if args.startswith("fac_huber"):
            fac_huber = float(args.split("=")[1])
            default_facs = False
        if args.startswith("fac_charbonnier"):
            fac_charbonnier = float(args.split("=")[1])
            default_facs = False

    if default_facs:
        fac_huber = 300
        fac_uv = 100
        fac_ms_ssim = 30
        fac_lpips = 10

    # obtain restore iteration info
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    
    if restore_iter == 0: # no need to get the last iteration if specified
        restore_iter = get_last_iter(model_dir, "LAN")

    num_train_iters += restore_iter

    print("The following parameters will be applied for training:")
    print("Path to the dataset: " + dataset_dir)
    print("Path to VGG-19 network: " + vgg_dir)
    print("Path to RGB data from DSLR: " + dslr_dir)
    print("Path to Raw data from phone: " + phone_dir)
    print("Path to Raw-to-RGB model network: " + model_dir)

    print("Restore Iteration: " + str(restore_iter))

    print("Batch size: " + str(batch_size))
    print("Training size: " + str(train_size))
    print("Learning rate: " + str(learning_rate))
    print("Evaluation step: " + str(eval_step))
    print("Training iterations: " + str(num_train_iters))

    print("Optimizer: " + optimizer)
    print("Loss function=" +
        " mse:" + str(fac_mse) +
        " l1:" + str(fac_l1) +
        " ssim:" + str(fac_ssim) +
        " ms-ssim:" + str(fac_ms_ssim) +
        " uv:" + str(fac_uv) +
        " vgg:" + str(fac_vgg) +
        " lpips:" + str(fac_lpips) +
        " huber:" + str(fac_huber) +
        " charbonnier:" + str(fac_charbonnier))
    return dataset_dir, model_dir, vgg_dir, dslr_dir, phone_dir, restore_iter,\
    patch_w, patch_h, batch_size, train_size, learning_rate, eval_step, num_train_iters, optimizer,\
    fac_mse, fac_l1, fac_ssim, fac_ms_ssim, fac_uv, fac_vgg, fac_lpips, fac_huber, fac_charbonnier


def process_test_model_args(arguments):
    # Specifying the default parameters for testing
    # --- data path ---
    dataset_dir = 'raw_images/'
    result_dir = None
    phone_dir = 'mediatek_raw_normal/'
    model_dir = 'models/'

    #--- model weights ---
    restore_iter = None

    # --- input size ---
    img_h = 3000 # default size
    img_w = 4000 # default size
    # --- more options ---
    use_gpu = True

    for args in arguments:
        # --- data path ---
        if args.startswith("dataset_dir"):
            dataset_dir = args.split("=")[1]
        if args.startswith("result_dir"):
            result_dir = args.split("=")[1]
        if args.startswith("phone_dir"):
            phone_dir = args.split("=")[1]
        if args.startswith("model_dir"):
            model_dir = args.split("=")[1]

        # --- model weights ---
        if args.startswith("restore_iter"):
            restore_iter = int(args.split("=")[1])

        # --- input size ---
        if args.startswith("img_h"):
            img_h = int(args.split("=")[1])
        if args.startswith("img_w"):
            img_w = int(args.split("=")[1])

        # --- more options ---        
        if args.startswith("use_gpu"):
            use_gpu = eval(args.split("=")[1])

    if result_dir is None:
        result_dir = model_dir

    # obtain restore iteration info (necessary if no pre-trained model or not random weights)
    if restore_iter is None: # need to restore a model
        restore_iter = get_last_iter(model_dir, "LAN")
        if restore_iter == -1:
            print("Error: Cannot find any pre-trained models for LAN")
            sys.exit()

    print("The following parameters will be applied for testing:")
    print("Path to the dataset: " + dataset_dir)
    print("Path to result images: " + result_dir)
    print("Path to Raw data from phone: " + phone_dir)
    print("Path to Raw-to-RGB model network: " + model_dir)
    print("Restore itearation" + str(restore_iter))

    return dataset_dir, result_dir, phone_dir, model_dir,\
    restore_iter, img_h, img_w, use_gpu


def process_evaluate_model_args(arguments):
        # Specifying the default parameters for numerical evaluation
    # --- data path ---
    dataset_dir = 'raw_images/'
    vgg_dir = 'vgg_pretrained/imagenet-vgg-verydeep-19.mat'
    dslr_dir = 'fujifilm/'
    phone_dir = 'mediatek_raw/'
    model_dir = 'models/'

    #--- model weights ---
    restore_iter = None

    # --- input size ---
    img_h = 256 # default size
    img_w = 256 # default size
    # --- more options ---
    use_gpu = True
    batch_size = 10

    for args in arguments:
        # --- data path ---
        if args.startswith("dataset_dir"):
            dataset_dir = args.split("=")[1]
        if args.startswith("vgg_dir"):
            vgg_dir = args.split("=")[1]
        if args.startswith("dslr_dir"):
            dslr_dir = args.split("=")[1]
        if args.startswith("phone_dir"):
            phone_dir = args.split("=")[1]
        if args.startswith("model_dir"):
            model_dir = args.split("=")[1]

        # --- model weights ---
        if args.startswith("restore_iter"):
            restore_iter = int(args.split("=")[1])

        # --- input size ---
        if args.startswith("img_h"):
            img_h = int(args.split("=")[1])
        if args.startswith("img_w"):
            img_w = int(args.split("=")[1])

        # --- more options ---        
        if args.startswith("use_gpu"):
            use_gpu = eval(args.split("=")[1])
        if args.startswith("batch_size"):
            batch_size = int(args.split("=")[1])

    # obtain restore iteration info (necessary if no pre-trained model or not random weights)
    if restore_iter is None: # need to restore a model
        restore_iter = get_last_iter(model_dir, "LAN")
        if restore_iter == -1:
            print("Error: Cannot find any pre-trained models for LAN")
            sys.exit()

    print("The following parameters will be applied for testing:")
    print("Path to the dataset: " + dataset_dir)
    print("Path to VGG-19 network: " + vgg_dir)
    print("Path to RGB data from DSLR: " + dslr_dir)
    print("Path to Raw data from phone: " + phone_dir)
    print("Path to Raw-to-RGB model network: " + model_dir)
    print("Restore iteration:" + str(restore_iter))
    print("Batch size: " + str(batch_size))

    return dataset_dir, vgg_dir, dslr_dir, phone_dir, model_dir,\
    restore_iter, img_h, img_w, batch_size, use_gpu

def get_last_iter(model_dir, name_model):
    saved_models = [int(model_file.split(".")[0].split("_")[-1])
                    for model_file in os.listdir(model_dir)
                    if model_file.startswith(name_model)]

    if len(saved_models) > 0:
        return np.max(saved_models)
    else:
        return 0


def log10(x):
  numerator = tf.math.log(x)
  denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
  return numerator / denominator

def gauss_kernel(kernlen=21, nsig=3, channels=1):
    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    out_filter = np.array(kernel, dtype = np.float32)
    out_filter = out_filter.reshape((kernlen, kernlen, 1, 1))
    out_filter = np.repeat(out_filter, channels, axis = 2)
    return out_filter

def blur(x):
    kernel_var = gauss_kernel(21, 3, 3)
    return tf.nn.depthwise_conv2d(x, kernel_var, [1, 1, 1, 1], padding='SAME')