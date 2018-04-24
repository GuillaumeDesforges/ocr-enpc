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
label = "Learning rate = 0.0001"
path = "C:\\Users\\teovi\\Documents\\ocropy"

os.chdir(path)

index = codecs.open('index.txt', 'r', encoding='utf-8')
data = codecs.open('results.txt', 'r', encoding='utf-8')

indexes=[]
error=[]
indexesref=[]
errorref=[]

for line in index:
    # print(line)
    indexes.append(int(line[14:-10]))  #Mettre -10 à la place de -11 parfois
    
for line in indexref:
    indexesref.append(int(line[13:-10]))  #Mettre -10 à la place de -11 parfois
    
for line in data:
    if "err " in line:
        error.append(float(line[-11:-3]))

for line in dataref:
    if "err " in line:
        errorref.append(float(line[-11:-3]))

print("L'erreur minimale est de "+ str(min(error)) +"%, atteinte pour " + str(indexes[np.argmin(error)]) + " itérations")
# print("L'erreur minimale est de "+ str(min(errorref)) +"%, atteinte pour " + str(indexesref[np.argmin(errorref)]) + " itérations pour la solution de référence")
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
    plt.xlim(min(min(indexes), min(indexes2)),105000)
    plt.legend(fontsize=30)
    plt.ylabel("Pourcentage d'erreurs sur les données de test", fontsize=30)
    plt.xlabel("Nombre d'itérations dans l'entraînement", fontsize=30)
    plt.grid(True)
    
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()
    
    # plt.savefig('error_plot_compare.png')

    
plt_1courbe(indexesref, errorref)
# plt_2courbes([indexesref[10]]+ indexes, [errorref[10]] + error, label,  indexesref, errorref, labelref)
index.close()
data.close()
