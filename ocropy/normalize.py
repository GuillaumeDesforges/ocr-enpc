# coding: unicode

import os
import codecs
import unicodedata

print("Script deprecated : path was hard set in the script, change it or do not use this script")
path = "C:\\Users\\teovi\\Documents\\ocropy\\book"
os.chdir(path)

compteur=0
strchar=""

for folder in os.listdir(path):
    for file in os.listdir(path+"\\"+folder):
        if "gt" in file:
            # print(folder+"\\"+file)
            f = codecs.open(folder+"\\"+file, 'r', encoding='utf-8')
            for line in f:
                newline = ""
                
                for char in line:
                    if char not in strchar:
                        strchar+=char
                    if char == "ſ":
                        newline+="Z"
                    else:
                        newline+=char
                        
                newline=unicodedata.normalize('NFKC',newline)
            f.close()
            # f = codecs.open(folder+"\\"+file, 'w', encoding='utf-8')
            # f.write(newline)
            # f.close()


strchar = ''.join(sorted(strchar))
print(strchar)

# '-./9ABCDFGJKLMNOPQRSTabcefghklmnopqrstuvwxyzãõ÷ıũɑ́̃͛ͣͤͥͦδ᷑ẽꝑꝓꝛꝯ﻿]
# '-./9ABCDFGJKLMNOPQRSTabcefghklmnopqrstuvwxyzãõ÷ıũɑ́̃͛ͣͤͥͦδ᷑ẽ⁹ꝑꝓꝛꝯ\ue476\uf217"

normstrchar1 = unicodedata.normalize('NFKC', strchar)
normstrchar1 = ''.join(sorted(normstrchar))

print(normstrchar1)

normstrchar2 = unicodedata.normalize('NFC', strchar)
normstrchar2 = ''.join(sorted(normstrchar))

print(normstrchar2)
