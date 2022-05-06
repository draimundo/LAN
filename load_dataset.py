###########################################
# Dataloader for training/validation data #
###########################################

from __future__ import print_function
from PIL import Image
import imageio
import os
import numpy as np
from tqdm import tqdm


def load_val_data(dataset_dir, dslr_dir, phone_dir, PATCH_WIDTH, PATCH_HEIGHT):
    val_directory_dslr = dataset_dir + 'val/' + dslr_dir
    val_directory_phone = dataset_dir + 'val/' + phone_dir
 
    PATCH_DEPTH = 1

    NUM_VAL_IMAGES = len([name for name in os.listdir(val_directory_phone)
                           if os.path.isfile(os.path.join(val_directory_phone, name))])

    val_data = np.zeros((NUM_VAL_IMAGES, PATCH_WIDTH, PATCH_HEIGHT, PATCH_DEPTH))
    val_answ = np.zeros((NUM_VAL_IMAGES, int(PATCH_WIDTH), int(PATCH_HEIGHT), 3))

    format_dslr = str.split(os.listdir(val_directory_dslr)[0],'.')[-1]

    for i in tqdm(range(0, NUM_VAL_IMAGES)):
        In = np.asarray(imageio.imread(val_directory_phone + str(i) + '.png'))
        val_data[i, ..., 0] = In

        I = Image.open(val_directory_dslr + str(i) + '.' + format_dslr)
        I = np.float32(np.reshape(I, [1, int(PATCH_WIDTH), int(PATCH_HEIGHT), 3])) / 255
        val_answ[i, :] = I

    return val_data, val_answ

def load_test_data(dataset_dir, dslr_dir, phone_dir, PATCH_WIDTH, PATCH_HEIGHT):
    test_directory_dslr = dataset_dir + 'test/' + dslr_dir
    test_directory_phone = dataset_dir + 'test/' + phone_dir

    PATCH_DEPTH = 1

    # NUM_VAL_IMAGES = 1204
    NUM_TEST_IMAGES = len([name for name in os.listdir(test_directory_phone)
                           if os.path.isfile(os.path.join(test_directory_phone, name))])

    test_data = np.zeros((NUM_TEST_IMAGES, PATCH_WIDTH, PATCH_HEIGHT, PATCH_DEPTH))
    test_answ = np.zeros((NUM_TEST_IMAGES, int(PATCH_WIDTH), int(PATCH_HEIGHT), 3))

    for i in tqdm(range(0, NUM_TEST_IMAGES)):
        In = np.asarray(imageio.imread(test_directory_phone + str(i) + '.png'))
        test_data[i, ..., 0] = In

        I = Image.open(test_directory_dslr + str(i) + '.png')
        I = np.float32(np.reshape(I, [1, int(PATCH_WIDTH), int(PATCH_HEIGHT), 3])) / 255
        test_answ[i, :] = I

    return test_data, test_answ


def load_train_patch(dataset_dir, dslr_dir, phone_dir, TRAIN_SIZE, PATCH_WIDTH, PATCH_HEIGHT):
    train_directory_dslr = dataset_dir + 'train/' + dslr_dir
    train_directory_phone = dataset_dir + 'train/' + phone_dir

    PATCH_DEPTH = 1
            
    # get the image format (e.g. 'png')
    format_dslr = str.split(os.listdir(train_directory_dslr)[0],'.')[-1]

    # determine training image numbers by listing all files in the folder
    NUM_TRAINING_IMAGES = len([name for name in os.listdir(train_directory_phone)
                               if os.path.isfile(os.path.join(train_directory_phone, name))])

    TRAIN_IMAGES = np.random.choice(np.arange(0, int(NUM_TRAINING_IMAGES)), TRAIN_SIZE, replace=False)

    train_data = np.zeros((TRAIN_SIZE, PATCH_WIDTH, PATCH_HEIGHT, PATCH_DEPTH))
    train_answ = np.zeros((TRAIN_SIZE, int(PATCH_WIDTH), int(PATCH_HEIGHT), 3))

    i = 0
    for img in tqdm(TRAIN_IMAGES):
        In = np.asarray(imageio.imread(train_directory_phone + str(img) + '.png'))
        train_data[i, ..., 0] = In

        I = Image.open(train_directory_dslr + str(img) + '.' + format_dslr)
        I = np.float32(np.reshape(I, [1, int(PATCH_WIDTH), int(PATCH_HEIGHT), 3])) / 255
        train_answ[i, :] = I

        i += 1

    return train_data, train_answ