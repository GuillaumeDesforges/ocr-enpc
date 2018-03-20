# coding: unicode

import numpy as np
import os
import codecs
import matplotlib.pyplot as plt

# path = "C:\\Users\\teovi\\Documents\\ocropy"
path = "C:\\Users\\teovi\\Documents\\IMI\Projet ML\ocr-enpc\ocropy\Models\Default_normalized"

os.chdir(path)

indexes=[]
error=[]

index = codecs.open('index.txt', 'r', encoding='utf-8')
data = codecs.open('results.txt', 'r', encoding='utf-8')

errors_dic={}
for line in index:
    indexes.append(int(line[13:-11]))  #Mettre -10 à la place de -11 parfois

print(indexes)
    
for line in data:
    if "err " in line:
        error.append(float(line[-11:-3]))

print("L'erreur minimale est de "+ str(min(error)) +"%, atteinte pour " + str(indexes[np.argmin(error)]) + " itérations")

plt.figure()
plt.plot(indexes,error)
plt.ylim(0, max(error))
plt.xlim(min(indexes),105000)
plt.legend(fontsize=100)
plt.ylabel("Pourcentage d'erreurs sur les données de test", fontsize=30)
plt.xlabel("Nombre d'itérations dans l'entraînement", fontsize=30)
plt.grid(True)

figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
plt.show()

plt.savefig('error_plot.png')

index.close()
data.close()
