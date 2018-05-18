import pickle
import os
import codecs
import gzip

##


# from __future__ import print_function

path = 'C:\\Users\\teovi\\Documents\\ocropy\\ocrolib'
os.chdir(path)


import os
import os.path
import re
import sys
import sysconfig
import unicodedata
import inspect
import glob
# import cPickle
from exceptions import (BadClassLabel, BadInput, FileNotFound,
                                OcropusException)

import numpy
from numpy import (amax, amin, array, bitwise_and, clip, dtype, mean, minimum,
                   nan, sin, sqrt, zeros)
import pylab
from pylab import (clf, cm, ginput, gray, imshow, ion, subplot, where)
from scipy.ndimage import morphology, measurements
import PIL

from default import getlocal
# from toplevel import (checks, ABINARY2, AINT2, AINT3, BOOL, DARKSEG, GRAYSCALE,
                      # LIGHTSEG, LINESEG, PAGESEG)
# import chars
import codecs
# import ligatures
# import lstm
# import morph
# import multiprocessing
# import sl

##
path = 'C:\\Users\\teovi\\Documents\\IMI\\Projet ML\\ocr-enpc\\ocropy'
os.chdir(path)

def load_object(fname):
    """Loads an object from disk. By default, this handles zipped files
    and searches in the usual places for OCRopus. It also handles some
    class names that have changed."""

    with gzip.open(fname, 'rb') as stream:
        unpickler = pickle.Unpickler(stream)
        return unpickler.load()

            

file = "test6-18.pyrnn.gz"

f = codecs.open(file, 'r', encoding='utf-8')
            
            
# for line in files:
    # print(line)
load_object(path + "\\" + file)