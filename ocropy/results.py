# coding: unicode

import numpy as np
import os
import codecs
import matplotlib.pyplot as plt


## First data

path = "C:\\Users\\teovi\\Documents\\IMI\\Projet Ocr\\Models\\hiddensize=500"

os.chdir(path)

label1 = "Learning rate = 0.0001"


data1 = codecs.open('results.txt', 'r', encoding='utf-8')
index1 = codecs.open('index.txt', 'r', encoding='utf-8')


indexes1 = []
error1 = []

for line in index1:
    indexes1.append(int(line[13:-10]))  #Mettre -10 à la place de -11 parfois
    
for line in data1:
    if "err " in line:
        error1.append(float(line[-11:-3]))
        
print(min(error1))
print(indexes1[np.argmin(error1)])

## Second data

path = "C:\\Users\\teovi\\Documents\\IMI\\Projet Ocr\\Models\\hiddensize=500\\train"
os.chdir(path)

label2 = "Learning rate = 0.00001"


data2 = codecs.open('results.txt', 'r', encoding='utf-8')
index2 = codecs.open('index.txt', 'r', encoding='utf-8')


indexes2 = []
error2 = []

for line in index2:
    indexes2.append(int(line[13:-10]))  #Mettre -10 à la place de -11 parfois
    
for line in data2:
    if "err " in line:
        error2.append(float(line[-11:-3]))

print(min(error2))
print(indexes2[np.argmin(error2)])

## Third data

path = "C:\\Users\\teovi\\Documents\\IMI\\Projet Ocr\\Models\\hiddensize=200\\0.00001"
os.chdir(path)

label3 = "Learning rate = 0.000001"


data3 = codecs.open('results.txt', 'r', encoding='utf-8')
index3 = codecs.open('index.txt', 'r', encoding='utf-8')


indexes3 = []
error3 = []

for line in index3:
    indexes3.append(int(line[13:-10]))  #Mettre -10 à la place de -11 parfois
    
for line in data3:
    if "err " in line:
        error3.append(float(line[-11:-3]))

# print(min(error3))
# print(indexes3[np.argmin(error3)])

## Plot functions

def plt_1courbe(indexes, error):
    plt.figure()
    line = plt.plot(indexes,error, linewidth=2)
    
    
    plt.ylim(0, 60)
    plt.xlim(min(indexes),max(indexes)+1000)
    plt.legend(fontsize=30)
    plt.ylabel("Pourcentage d'erreurs sur les données de test", fontsize=30)
    plt.xlabel("Nombre d'itérations dans l'entraînement", fontsize=30)
    plt.grid(True)
    
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()
    
    # plt.savefig('error_plot_.png')

def plt_2courbes(indexes1, error1, label1, indexes2, error2, label2):
    plt.figure()
    line1 = plt.plot(indexes1,error1, label = label1, linewidth=2)
    legend1 = plt.legend(handles=line1, loc=1)
    
    line2 = plt.plot(indexes2,error2, label=label2, linewidth=2)
    legend2 = plt.legend(handles=line2, loc=1)
    
    plt.ylim(0, 60)
    # plt.xlim(1000, 50000)
    plt.legend(fontsize=30)
    plt.ylabel("Pourcentage d'erreurs sur les données de test", fontsize=30)
    plt.xlabel("Nombre d'itérations dans l'entraînement", fontsize=30)
    plt.grid(True)
    
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()
    
    # plt.savefig('error_plot_compare.png')


def plt_3courbes(indexes1, error1, label1, indexes2, error2, label2, indexes3, error3, label3):
    plt.figure()
    line1 = plt.plot(indexes1,error1, label = label1, linewidth=2)
    legend1 = plt.legend(handles=line1, loc=1)
    
    line2 = plt.plot(indexes2,error2, label=label2, linewidth=2)
    legend2 = plt.legend(handles=line2, loc=1)
    
    line3 = plt.plot(indexes3,error3, label=label3, linewidth=2)
    legend3 = plt.legend(handles=line2, loc=1)
    
    plt.ylim(0, max(max(error1),max(error2), max(error3)))
    plt.xlim(0,150000)
    plt.legend(fontsize=30)
    plt.ylabel("Pourcentage d'erreurs sur les données de test", fontsize=30)
    plt.xlabel("Nombre d'itérations dans l'entraînement", fontsize=30)
    plt.grid(True)
    
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()
 

## Calls


# plt_1courbe(indexes1, error1)
# plt_2courbes(indexes1[:36], error1[:36], label1,  indexes2, error2, label2)
# plt_3courbes(indexes1[:95], error1[:95], label1,  indexes2, error2, label2, indexes3, error3, label3 )


# print(indexes)
# print(indexes2)

##

X = [8.86, 7.55, 7.64]
X2 = [2.90, 1.75, 1.6]
Y = [100, 200, 500]
Ylog = np.log(np.array([100, 200, 500]))
plt.figure()
plt.grid(True)
plt.plot(Y,X, label = "Test data")
plt.plot(Y,X2, label= "Train data")
plt.title("Error in function of the size of the LSTM")
plt.grid(True)
plt.ylim(0,10)
plt.legend()
plt.show()