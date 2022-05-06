##################################################
# Run the trained model on visual testing images #
##################################################

import numpy as np
import tensorflow as tf
import imageio
import sys
import os
import rawpy

from model import lan_g

from tqdm import tqdm
from datetime import datetime

import utils

dataset_dir, result_dir, phone_dir, model_dir,\
    restore_iter, img_h, img_w, use_gpu = utils.process_test_model_args(sys.argv)

IMAGE_HEIGHT, IMAGE_WIDTH = img_h, img_w
TARGET_DEPTH = 3

FAC_PATCH = 1
PATCH_DEPTH = 1

TARGET_HEIGHT = IMAGE_HEIGHT
TARGET_WIDTH = IMAGE_WIDTH

PATCH_HEIGHT = int(np.floor(IMAGE_HEIGHT//FAC_PATCH))
PATCH_WIDTH = int(np.floor(IMAGE_WIDTH//FAC_PATCH))


# Disable gpu if specified
config = tf.compat.v1.ConfigProto(device_count={'GPU': 0}) if not use_gpu else None

if not os.path.isdir("results/full-resolution/"+ result_dir):
    os.makedirs("results/full-resolution/"+ result_dir, exist_ok=True)

with tf.compat.v1.Session(config=config) as sess:
    time_start = datetime.now()

    # Placeholders for test data
    x_ = tf.compat.v1.placeholder(tf.float32, [1, PATCH_HEIGHT, PATCH_WIDTH, PATCH_DEPTH])


    # generate enhanced image
    # Get the processed enhanced image
    enhanced = lan_g(x_)


    # Determine model weights
    saver = tf.compat.v1.train.Saver()

    # Processing full-resolution RAW images
    test_dir_full = 'validation_full_resolution_visual_data/' + phone_dir

    test_photos = [f for f in os.listdir(test_dir_full) if os.path.isfile(test_dir_full + f)]
    test_photos.sort()

    print("Loading images")
    images = np.zeros((len(test_photos), PATCH_HEIGHT, PATCH_WIDTH, PATCH_DEPTH))
    for i, photo in tqdm(enumerate(test_photos)):
        print("Processing image " + photo)

        In = np.asarray(rawpy.imread((test_dir_full + photo)).raw_image.astype(np.float32))
        images[i,..., 0] = In[0:PATCH_HEIGHT, 0:PATCH_WIDTH, ...]
                
    print("Images loaded")
    # Run inference


    saver.restore(sess, model_dir + "LAN_iteration_" + str(restore_iter) + ".ckpt")
    
    for i, photo in enumerate(test_photos):
        enhanced_tensor = sess.run(enhanced, feed_dict={x_: [images[i,...]]})
        enhanced_image = np.reshape(enhanced_tensor, [TARGET_HEIGHT, TARGET_WIDTH, TARGET_DEPTH])

        # Save the results as .png images
        photo_name = photo.rsplit(".", 1)[0]
        imageio.imwrite("results/full-resolution/"+ result_dir + photo_name +
                    "_iteration_" + str(restore_iter) + ".png", enhanced_image)