##################################################
# Train a RAW-to-RGB model using training images #
##################################################

import tensorflow as tf
import numpy as np
import sys
from datetime import datetime

from load_dataset import load_train_patch, load_val_data
from model import lan_g
import utils
import vgg
import lpips_tf

from tqdm import tqdm

from RAdam import RAdamOptimizer


# Processing command arguments
dataset_dir, model_dir, vgg_dir, dslr_dir, phone_dir, restore_iter,\
    patch_w, patch_h, batch_size, train_size, learning_rate, eval_step, num_train_iters, optimizer,\
    fac_mse, fac_l1, fac_ssim, fac_ms_ssim, fac_uv, fac_vgg, fac_lpips, fac_huber, fac_charbonnier \
    = utils.process_command_args(sys.argv)

# Defining the size of the input and target image patches

FAC_PATCH = 1
PATCH_DEPTH = 1

PATCH_WIDTH = patch_w//FAC_PATCH
PATCH_HEIGHT = patch_h//FAC_PATCH
PATCH_SIZE = PATCH_WIDTH * PATCH_HEIGHT * 3

TARGET_WIDTH = int(PATCH_WIDTH * FAC_PATCH)
TARGET_HEIGHT = int(PATCH_HEIGHT * FAC_PATCH)
TARGET_DEPTH = 3
TARGET_SIZE = TARGET_WIDTH * TARGET_HEIGHT * TARGET_DEPTH

np.random.seed(0)
tf.random.set_seed(0)

# Defining the model architecture
with tf.Graph().as_default(), tf.compat.v1.Session() as sess:
    time_start = datetime.now()

    # Placeholders for training data
    phone_ = tf.compat.v1.placeholder(tf.float32, [batch_size, PATCH_HEIGHT, PATCH_WIDTH, PATCH_DEPTH])
    dslr_ = tf.compat.v1.placeholder(tf.float32, [batch_size, TARGET_HEIGHT, TARGET_WIDTH, TARGET_DEPTH])

    # Get the processed enhanced image
    enhanced = lan_g(phone_)

    print("Num variables:" + str(np.sum([np.prod(v.get_shape().as_list()) for v in tf.compat.v1.trainable_variables()])))
    # Losses
    dslr_gray = tf.image.rgb_to_grayscale(dslr_)
    enhanced_gray = tf.image.rgb_to_grayscale(enhanced)

    # MSE loss
    loss_mse = tf.reduce_mean(tf.math.squared_difference(enhanced, dslr_))
    loss_generator = loss_mse * fac_mse
    loss_list = [loss_mse]
    loss_text = ["loss_mse"]

    # L1 loss
    loss_l1 = tf.reduce_mean(tf.abs(tf.math.subtract(enhanced, dslr_)))
    if fac_l1 > 0:
        loss_list.append(loss_l1)
        loss_text.append("loss_l1")
        loss_generator += loss_l1 * fac_l1

    eps = 1e-6
    loss_charbonnier = tf.reduce_mean(tf.sqrt(tf.math.squared_difference(enhanced, dslr_) + eps))
    if fac_charbonnier > 0:
        loss_list.append(loss_charbonnier)
        loss_text.append("loss_charbonnier")
        loss_generator += loss_charbonnier * fac_charbonnier

    # PSNR metric
    metric_psnr = tf.reduce_mean(tf.image.psnr(enhanced, dslr_, 1.0))
    loss_list.append(metric_psnr)
    loss_text.append("metric_psnr")

    # SSIM loss
    loss_ssim = 1 - tf.reduce_mean(tf.image.ssim(enhanced_gray, dslr_gray, 1.0))
    if fac_ssim > 0:
        loss_generator += loss_ssim * fac_ssim
        loss_list.append(loss_ssim)
        loss_text.append("loss_ssim")

    # MS-SSIM loss
    loss_ms_ssim = 1 - tf.reduce_mean(tf.image.ssim_multiscale(enhanced_gray, dslr_gray, 1.0))
    if fac_ms_ssim > 0:
        loss_generator += loss_ms_ssim * fac_ms_ssim
        loss_list.append(loss_ms_ssim)
        loss_text.append("loss_ms_ssim")

    ## UV loss
    dslr_yuv = tf.image.rgb_to_yuv(dslr_)
    enhanced_lab = tf.image.rgb_to_yuv(enhanced)
    enhanced_uv_blur = utils.blur(enhanced_lab)[..., -2:]
    dslr_uv_blur = utils.blur(dslr_yuv)[..., -2:]
    loss_uv = tf.reduce_mean(tf.abs(tf.math.subtract(dslr_uv_blur, enhanced_uv_blur)))
    if fac_uv > 0:
        loss_generator += loss_uv * fac_uv
        loss_list.append(loss_uv)
        loss_text.append("loss_uv")

    # Huber loss
    delta = 1
    abs_error = tf.abs(tf.math.subtract(enhanced, dslr_))
    quadratic = tf.math.minimum(abs_error, delta)
    linear = tf.math.subtract(abs_error, quadratic)
    loss_huber = tf.reduce_mean(0.5*tf.math.square(quadratic)+linear)
    if fac_huber > 0:
        loss_generator += loss_huber * fac_huber
        loss_list.append(loss_huber)
        loss_text.append("loss_huber")

    # Content loss
    CONTENT_LAYER = 'relu5_4'
    enhanced_vgg = vgg.net(vgg_dir, vgg.preprocess(enhanced * 255))
    dslr_vgg = vgg.net(vgg_dir, vgg.preprocess(dslr_ * 255))
    loss_vgg = tf.reduce_mean(tf.math.squared_difference(enhanced_vgg[CONTENT_LAYER], dslr_vgg[CONTENT_LAYER]))
    loss_list.append(loss_vgg)
    loss_text.append("loss_vgg")
    if fac_vgg > 0:
        loss_generator += loss_vgg * fac_vgg

    ## LPIPS
    loss_lpips = tf.reduce_mean(lpips_tf.lpips(enhanced, dslr_, net='alex'))
    loss_list.append(loss_lpips)
    loss_text.append("loss_lpips")
    if fac_lpips > 0:
        loss_generator += loss_lpips * fac_lpips


    ## Final loss function
    loss_list.insert(0, loss_generator)
    loss_text.insert(0, "loss_generator")


    # Optimize network parameters
    vars_lan_g = [v for v in tf.compat.v1.global_variables() if v.name.startswith("generator")]
    if optimizer == "radam":
        train_step_lan_g = RAdamOptimizer(learning_rate=learning_rate).minimize(loss_generator, var_list=vars_lan_g)
    elif optimizer == "adam":
        train_step_lan_g = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(loss_generator, var_list=vars_lan_g)
    else:
        print("Optimizer not found -> using Adam")
        train_step_lan_g = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(loss_generator, var_list=vars_lan_g)

    # Initialize and restore the variables
    print("Initializing variables...")
    sess.run(tf.compat.v1.global_variables_initializer())

    saver = tf.compat.v1.train.Saver(var_list=vars_lan_g, max_to_keep=1000)

    if restore_iter > 0: # restore the variables/weights
        name_model_restore_full = "lan" + "_iteration_" + str(restore_iter)
        print("Restoring Variables from:", name_model_restore_full)
        saver.restore(sess, model_dir + name_model_restore_full + ".ckpt")

    # Loading training and validation data
    print("Loading validation data...")
    val_data, val_answ = load_val_data(dataset_dir, dslr_dir, phone_dir, PATCH_WIDTH, PATCH_HEIGHT)
    print("Validation data was loaded\n")

    print("Loading training data...")
    train_data, train_answ = load_train_patch(dataset_dir, dslr_dir, phone_dir, train_size, PATCH_WIDTH, PATCH_HEIGHT)
    print("Training data was loaded\n")

    VAL_SIZE = val_data.shape[0]
    num_val_batches = int(val_data.shape[0] / batch_size)

    print("Training network...")

    iter_start = restore_iter+1 if restore_iter > 0 else 0
    logs = open(model_dir + "logs_" + str(iter_start) + "-" + str(num_train_iters) + ".txt", "w+")
    logs.close()

    loss_lan_g_ = 0.0
    
    for i in tqdm(range(iter_start, num_train_iters + 1), miniters=100):
        # Train generator
        idx_g = np.random.randint(0, train_size, batch_size)
        phone_g = train_data[idx_g]
        dslr_g = train_answ[idx_g]

        feed_g = {phone_: phone_g, dslr_: dslr_g}
        [loss_temp, temp] = sess.run([loss_generator, train_step_lan_g], feed_dict=feed_g)
        loss_lan_g_ += loss_temp / eval_step

        #  Evaluate model
        if i % eval_step == 0:
            val_losses_g = np.zeros((1, len(loss_text)))

            for j in range(num_val_batches):
                be = j * batch_size
                en = (j+1) * batch_size

                phone_images = val_data[be:en]
                dslr_images = val_answ[be:en]

                valdict = {phone_: phone_images, dslr_: dslr_images}
                toRun = [loss_list]

                loss_temp = sess.run(toRun, feed_dict=valdict)
                val_losses_g += np.asarray(loss_temp) / num_val_batches

            logs_gen = "step %d | training: %.4g,  "  % (i, loss_lan_g_)
            for idx, loss in enumerate(loss_text):
                logs_gen += "%s: %.4g; " % (loss, val_losses_g[0][idx])
            print(logs_gen)

            # Save the results to log file
            logs = open(model_dir + "logs_" + str(iter_start) + "-" + str(num_train_iters) + ".txt", "a")
            logs.write(logs_gen)
            logs.write('\n')
            logs.close()

            # Saving the model that corresponds to the current iteration
            saver.save(sess, model_dir + "LAN_iteration_" + str(i) + ".ckpt", write_meta_graph=False)

            loss_lan_g_ = 0.0

        # Loading new training data
        if i % 1000 == 0  and i > 0:
            del train_data
            del train_answ
            train_data, train_answ = load_train_patch(dataset_dir, dslr_dir, phone_dir, train_size, PATCH_WIDTH, PATCH_HEIGHT)
    print('total train/eval time:', datetime.now() - time_start)