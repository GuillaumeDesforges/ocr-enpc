# coding: unicode

import numpy as np
import os
path = "C:\\Users\\teovi\\Documents\\ocropy"
import codecs
import unicodedata
import matplotlib.pyplot as plt
os.chdir(path)

indexes=[]
error=[]

index = codecs.open('index.txt', 'r', encoding='utf-8')
data = codecs.open('results.txt', 'r', encoding='utf-8')

errors_dic={}
for line in index:
    indexes.append(int(line[12:-10]))

print(indexes)
    
for line in data:
    if "err " in line:
        error.append(float(line[-11:-3]))

print("L'erreur minimale est de "+ str(min(error)) +"%, atteinte pour " + str(indexes[np.argmin(error)]) + " itérations")

plt.figure()
plt.plot(indexes,error)
plt.ylim(0, 80)
plt.xlim(3000,max(indexes))
plt.legend(fontsize=100)
plt.ylabel("Pourcentage d'erreurs sur les données de test", fontsize=30)
plt.xlabel("Nombre d'itérations dans l'entraînement", fontsize=30)
plt.grid(True)
plt.show()

index.close()
data.close()