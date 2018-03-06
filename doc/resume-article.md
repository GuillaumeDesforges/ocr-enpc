# Resume article

Petit résumer de l'article : http://graal.hypotheses.org/786

## En bref
OCRpy marche bien pour reconnaître les écrits imprimés, on cherche à reconnaître les écrits manuscrites

## Etapes
* Création du set d'entrainement
    * On enregistre des images dans un dossier (ici /tif)
    * On extrait les colonnes puis les lignes
        * Automatisation avec `ocropus-gpageseg` : ça ne marche pas
        * Finalement fait à la main
    * Annotation avec `ocropus-gtedit`
        * Résultats commencent à 400 lignes traduites
        * Dans la littérature on parle de 1000 à 5000 lignes... (q_q)
        * On peut entrainer avec 400, utiliser le modèle pour préannoter plus d'entrée et augementer récursivement le dataset, et donc la performance
* Entraînement de OCRpy
    * Extraire les données annotées : `ocropus-gtedit extract`
    * On n'oublie pas de split train/test
    * Entraînement : ocropus-rtrain
    * Combien d'itérations ?
        * Littérature : de 30000 à 200000 itérations (relatif au nombre de lignes/images sur lesquelles on entraîne
        * Une *époque* correspond à un cylce dans le dataset, environ 100 époques conseillées
* Evaluation
    * Regarder l'évolution du taux d'erreur
    * `ocropus-econf` pour voir les erreures communes
* Application sur tout le manuscrit
    * "predict"
    * `ocropus-rpred`
* Peut-on appliquer sur un autre manuscrit ?
    * Beaucoup plus efficace de réentrainer un nouveau modèle
