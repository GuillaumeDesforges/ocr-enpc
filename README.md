# Machine Learning Project

**Guillaume DESFORGES & Théo Viel**

Dans ce projet, nous nous intéressons à la compréhension des aspects théoriques et à l'implémentation d'un OCR.

Un OCR est un système de reconnaissance de caractères automatique.
Il existe déjà plusieurs méthodes pour cette tâche, la plus récente étant une méthode de *Deep Learning*.

## Informations sur le répo :

* Le rapport et les slides se trouvent dans le dossier *doc*.
* Les scripts Python utilisés et les courbes obtenues dans le cadre de l'étude d'ocropy se trouvent dans le dossier *ocropy*
* Le notebook du dossier *neuralnetworks* peut être ouvert en slideshow, et contient une introduction à Keras pour notre problème.

## Contenu :

Le rapport contient principalement une explication plus détaillée de l'utilisation d'ocropy.
Pour l'utilisation de Keras, se référer au notebook.
La théorie sur la ctc-loss est les RNN est détaillée dans les slides, se référer à la doc pour plus de précisions.

Enfin, l'app PyQt4 a été développée dans un autre repository : 
https://github.com/GuillaumeDesforges/simple-ocr

Nous avons principalement utilisé comme données d'entraînement les manuscrits de la *Chanson d'Otinel* de Bodmer, qui nous avaient été fournis.

## Bibliographie

* Hannun, "Sequence Modeling with CTC", Distill, 2017. https://distill.pub/2017/ctc/
* A. Graves, S. Fernandez, F. Gomez, J. Schmidhuber, "Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks", 2006
* J-B. Camps, "Homemade manuscript OCR (1): OCRopy", Sacré Gr@@l, 2017, https://graal.hypotheses.org/786
* C. Olah, "Understanding LSTM Networks", 2015, http://colah.github.io/posts/2015-08-Understanding-LSTMs/
