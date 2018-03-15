#!/usr/bin/python

import os
import argparse

# TODO : arg parse book data folder path

DATA_FOLDER='../data/books/bodmer/full'

files = os.listdir(DATA_FOLDER)
files = [f for f in sorted(files) if f.endswith('.txt')]

charset = []
for file_name in files:
    with open(os.path.join(DATA_FOLDER, file_name), 'r') as f:
        for char in f.readline().rstrip():
            if char not in charset:
                print("Added {} from {}".format(char, file_name))
                charset.append(char)

print("Size of charset :", len(charset))
print("Charset : ", ' '.join(charset))
