#!/usr/bin/python

import sys
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from matplotlib.pyplot import imread

import cv2

import argparse

import Levenshtein

# ARGS

# command line arguments
parser = argparse.ArgumentParser(
        description='Learn and use an OCR with keras')

parser.add_argument(
        'mode',
        choices=['train', 'test', 'predict'])

parser.add_argument(
        'model_path',
        type=str,
        help='path to model')

parser.add_argument(
        'dataset_path',
        type=str,
        help='path to dataset folder with images and labels')

parser.add_argument(
        '-e', '--epochs',
        type=int,
        default=1,
        help='number of epochs to train on')

parser.add_argument(
        '-v', '--verbose',
        action='count',
        default=0,
        help='verbose')

# read args
cmd_args = parser.parse_args()

# read mode
mode = cmd_args.mode

# read model path
model_path = os.path.abspath(cmd_args.model_path)

# read dataset path
dataset_path = os.path.abspath(cmd_args.dataset_path)

# number of epochs
if mode == 'train':
    n_epochs = cmd_args.epochs

# read verbosity level
verbose = cmd_args.verbose
VERBOSE_LOW = 1
VERBOSE_HIGH = 2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(min(verbose - 1, 2))

# imports after verbosity set
import keras
from keras import backend as K
from keras import Model
from keras.layers import Input, Conv2D, MaxPooling2D, GRU, LSTM, Dense, Activation, Lambda, Reshape, Concatenate
from keras.optimizers import SGD


# TAGS (file paths)

# load folder data by walking dataset path
tags_img = []
tags_txt = []
for root, dirs, files in os.walk(dataset_path, topdown=False):
    root = os.path.abspath(root)
    for name in files:
        file_path = os.path.join(root, name)
        # store png and txt file path to corresponding storage, without file extension
        if file_path.endswith('.png'):
            tags_img.append(file_path.strip('.png'))
            if verbose >= VERBOSE_HIGH:
                print(file_path)
        if name.endswith('.txt'):
            tags_txt.append(file_path.strip('.txt'))
            if verbose >= VERBOSE_HIGH:
                print(file_path)

# merge tags where there is both img and txt
tags = [tag for tag in tags_img if tag in tags_txt]
if verbose:
    print("Found {} samples to load".format(len(tags)))


# TEXT LABELS

# read dataset text labels from files
Y_text = []
for i, tag in enumerate(tags):
    with open(tag+'.txt', 'r') as f:
        y_text = f.readline().rstrip()
        Y_text.append(y_text)

# keep only data with non empty label
tags, Y_text = map(list, zip(*filter(lambda x: x[1] != '', zip(tags, Y_text))))


# ALPHABET

# alphabet expected path
alphabet_path = os.path.join(model_path, 'alphabet.txt')

# not in train mode : load alphabet
if mode != 'train':
    if os.path.isfile(alphabet_path):
        with open(alphabet_path, 'r') as f:
            alphabet = f.readline().rstrip()
        if verbose:
            print("Alphabet loaded")
    else:
        print("No alphabet found in model folder : you first need to run train mode to make a model")
        sys.exit(1)
# train mode : make alphabet
else:
    # helper function
    def update_charset(Y_text):
        charset = {}
        for y_text in Y_text:
            for char in y_text:
                if char not in charset:
                    charset[char] = 0
                charset[char] += 1
        alphabet = ''.join(map(lambda x:x[0], sorted(charset.items(), key=lambda x:x[1], reverse=True)))
        return charset, alphabet
    # min occurences for a char to be taken into account
    char_n_threshold = 2
    # build the charset and the alphabet
    charset, alphabet = update_charset(Y_text)
    # until all chars in alphabet appear at least char_n_threshold times,
    # remove all data in dataset containing rare chars
    while any([char_n < char_n_threshold for char_n in charset.values()]):
        chars_to_delete = [char for char, char_n in charset.items() if char_n < char_n_threshold]
        tags, Y_text = map(list, zip(*filter(lambda x: not any([char_to_delete in x[1] for char_to_delete in chars_to_delete]), zip(tags, Y_text))))
        charset, alphabet = update_charset(Y_text)
    if verbose:
        print("Alphabet built")
    # save charset
    with open(alphabet_path, 'w') as f:
        f.write(alphabet)
    if verbose:
        print("Alphabet saved")
    
    if verbose:
        print("{} samples kept for training".format(len(tags)))

# CODED LABELS

# transform text label into code label
Y = [[alphabet.find(c) for c in y_text] for y_text in Y_text]
if verbose:
    print("Transformed text label to encoded label")


# IMAGES

# load images
imgs = [imread(tag+'.png') for tag in tags]
if verbose:
    print("Images read")

# change all images height as to fit fixed size
img_height = 150

X = []
for i, (tag, img) in enumerate(zip(tags, imgs)):
    # grayscale
    x = img[:, :, :-2].mean(axis=2) if len(img.shape) > 2 else img
    # current image shape
    img_height_in, img_width_in = x.shape
    # expected image shape
    img_width_out = int(img_width_in * (img_height/img_height_in))
    # open cv is (width, height)
    x = x.transpose()
    # resize
    x_resized = cv2.resize(x, dsize=(img_height, img_width_out), interpolation=cv2.INTER_LINEAR)
    # flip back to (height, width)
    x_resized = x_resized.transpose()
    # store result
    X.append(x_resized)


# WIDTHS

# image widths
X_widths = [x.shape[1] for x in X]
X_widths_max = max(X_widths)

# label widths
Y_lens = [len(y_text) for y_text in Y_text]
Y_lens_max = max(Y_lens)


# NUMPY & KERAS DATA FORMATING

# stack images
X = np.array([np.hstack([x, np.ones((img_height, X_widths_max-width))]) for x, width in zip(X, X_widths)])
# images are 
if K.image_data_format() == 'channels_first':
    X = np.swapaxes(X[:, :, :, np.newaxis], 1, 3)
    input_shape = (1, X_widths_max, img_height)
else:
    X = np.swapaxes(X, 1, 2)[:, :, :, np.newaxis]
    input_shape = (X_widths_max, img_height, 1)

# image widths array
X_widths = np.array(X_widths)

# labels array
Y = np.array([np.hstack([y, len(alphabet)*np.ones((Y_lens_max-l))]) for y, l in zip(Y, Y_lens)], dtype=np.int)

# label widths
Y_lens = np.array(Y_lens)


# MODEL

# images
inputs = Input(name='inputs', shape=input_shape, dtype='float32')
inputs_width = Input(name='inputs_width', shape=[1], dtype='int64')

# labels
labels = Input(name='labels', shape=[Y_lens_max], dtype='float32')
labels_length = Input(name='labels_length', shape=[1], dtype='int64')

# cnn
# conv1 = Conv2D(16, (3, 3), name='conv1', padding='same', activation='relu')(inputs)
# conv2 = Conv2D(32, (3, 3), name='conv2', padding='same', activation='relu')(conv1)
# conv3 = Conv2D(64, (3, 3), name='conv3', padding='same', activation='relu')(conv2)
# maxpool1 = MaxPooling2D(2, name='maxpool1')(conv3)

#rnn
n_features = img_height
rnn_inputs = inputs
reshape1 = Reshape((-1, n_features), name='reshape1')(rnn_inputs)
lstm1_f = LSTM(100, name='lstm1_f', return_sequences=True, kernel_initializer='he_normal')(reshape1)
lstm1_b = LSTM(100, name='lstm1_b', return_sequences=True, kernel_initializer='he_normal', go_backwards=True)(reshape1)
lstm1 = Concatenate(name='lstm1')([lstm1_f, lstm1_b])
rnn_out = lstm1

# char activation
dense1 = Dense(len(alphabet)+1, name='dense1')(rnn_out)
softmax1 = Activation('softmax', name='softmax1')(dense1)

# ctc loss
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    # y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

# wrapped ctc loss in lambda layer
ctc1 = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc1')([softmax1, labels, inputs_width, labels_length])

# model
if mode == 'train':
    model = Model(inputs=[inputs, labels, inputs_width, labels_length], outputs=ctc1)
    # note : clipnorm seems to speeds up convergence
    sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
    # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
    model.compile(
        loss={'ctc1': lambda y_true, y_pred: y_pred},
        optimizer=sgd)
else:
    model = Model(inputs, softmax1)

if verbose:
    model.summary()

# load existing model weights
model_weights_path = os.path.join(model_path, 'weights.h5')

if os.path.isfile(model_weights_path):
    if verbose >= VERBOSE_HIGH:
        print("Weights file found")
    model.load_weights(model_weights_path)
    if verbose:
        print("Weights loaded from file")
else:
    print("No weight file found")
    if mode != 'train':
        print("A trained model is needed for mode {}".format(mode))
        sys.exit(1)


# TRAIN

if mode == 'train':
    callback_tensorboard = keras.callbacks.TensorBoard(
        #log_dir='./logs',
        #histogram_freq=0,
        #write_graph=True,
        #write_grads=False,
        write_images=True,
        #embeddings_freq=0,
        #embeddings_layer_names=None,
        #embeddings_metadata=None
    )
    model.fit(
        x={
            'inputs': X,
            'inputs_width': X_widths,
            'labels': Y,
            'labels_length': Y_lens,
        },
        y=Y,
        epochs=n_epochs,
        callbacks=[callback_tensorboard],
    )
    print("Training done")
    model.save_weights(model_weights_path)
    if verbose:
        print("Saved weights")


# helper to decode
def pred_to_text(y_pred):
    text0 = ''.join([alphabet[l] if l != len(alphabet) else '-' for l in y_pred.tolist()])
    text1 = '-'
    for t in text0:
        if text1[-1] == t:
            continue
        text1 += t
    text2 = ''.join(filter(lambda x: x != '-', text1))
    return text2
 
# TEST
if mode == 'test':
    # pass forward
    Y_pred = model.predict(X).argmax(axis=2)
   
    # pred text
    Y_pred_text = [pred_to_text(y_pred) for y_pred in Y_pred]
    
    # distances
    Y_dist = [Levenshtein.distance(y_text, y_pred_text) for y_text, y_pred_text in zip(Y_text, Y_pred_text)]
    
    total_dist_diff = sum(Y_dist)
    total_text_len = sum([len(y_text) for y_text in Y_text])
    error_rate = total_dist_diff / total_text_len

    print("Test error rate : {:3.0f}%".format(error_rate*100))

# PREDICT
if mode == 'predict':
    # pass forward
    Y_pred = model.predict(X).argmax(axis=2)
   
    # pred text
    Y_pred_text = [pred_to_text(y_pred) for y_pred in Y_pred]
    
    for tag, y_pred_text in zip(tags, Y_pred_text):
        print(tag, y_pred_text)
