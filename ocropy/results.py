# coding: unicode

import numpy as np
import os
import codecs
import matplotlib.pyplot as plt


pathref = "C:\\Users\\teovi\\Documents\\IMI\Projet ML\ocr-enpc\ocropy\Models\Default"
labelref = "Learning rate = 0.001"
os.chdir(pathref)

dataref = codecs.open('results.txt', 'r', encoding='utf-8')
indexref= codecs.open('index.txt', 'r', encoding='utf-8')

# path = "C:\\Users\\teovi\\Documents\\IMI\Projet ML\ocr-enpc\ocropy\Models\\Default_n_hiddensize=200"
label = "Learning rate = 0.00001"
path = "C:\\Users\\teovi\\Documents\\ocropy"

os.chdir(path)

index = codecs.open('index.txt', 'r', encoding='utf-8')
data = codecs.open('results.txt', 'r', encoding='utf-8')


path2 = "C:\\Users\\teovi\\Documents\\IMI\\Projet ML\\Models\\adapt_lrate\\0.0001"
label2 = "Learning rate = 0.0001"
os.chdir(path2)

data2 = codecs.open('results.txt', 'r', encoding='utf-8')
index2= codecs.open('index.txt', 'r', encoding='utf-8')

indexes=[]
error=[]

indexesref=[]
errorref=[]

indexes2=[]
error2=[]

for line in index:
    indexes.append(int(line[14:-10]))  #Mettre -10 à la place de -11 parfois
    
for line in indexref:
    indexesref.append(int(line[13:-10]))  #Mettre -10 à la place de -11 parfois

for line in index2:
    indexes2.append(int(line[13:-10]))  #Mettre -10 à la place de -11 parfois
    

for line in data:
    if "err " in line:
        error.append(float(line[-11:-3]))

for line in data2:
    if "err " in line:
        error2.append(float(line[-11:-3]))

for line in dataref:
    if "err " in line:
        errorref.append(float(line[-11:-3]))

# print(error2)
# print(indexes2)
print("L'erreur minimale est de "+ str(min(error)) +"%, atteinte pour " + str(indexes[np.argmin(error)]) + " itérations")

def plt_1courbe(indexes, error):
    plt.figure()
    line = plt.plot(indexes,error, linewidth=2)
    
    
    plt.ylim(0, max(error))
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
    
    plt.ylim(0, max(max(error1),max(error2)))
    # plt.xlim(min(min(indexes), min(indexes2)),105000)
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
    plt.xlim(0,120000)
    plt.legend(fontsize=30)
    plt.ylabel("Pourcentage d'erreurs sur les données de test", fontsize=30)
    plt.xlabel("Nombre d'itérations dans l'entraînement", fontsize=30)
    plt.grid(True)
    
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()
 
    
# plt_1courbe(indexesref, errorref)
# plt_2courbes(indexes2, error2, label2,  indexesref[:49], errorref[:49], labelref)
plt_3courbes(indexesref[:49], errorref[:49], labelref, indexes2[:78-48], error2[:78-48], label2, indexes, error, label)
index.close()
data.close()

# print(indexes)
# print(indexes2)
# print(indexesref)
