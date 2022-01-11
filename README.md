# IA Project orientation and mood detection

## Introduction 

L’objectif  de  ce  projet  est  d’implémenter  une  application  de  reconnaissance  faciale  capable  de distinguer l’émotion et l’orientation du visage, mais également de détecter un visage masqué ou non. Ce projet étant une  ́evaluation, certaines contraintes étaient imposés. 

## Prérequis

Dans un premier temps il vous faudra installer Python et les librairies suivantes:
* tensorflow>=2.5.0
* keras==2.4.3
* imutils==0.5.4
* opencv-python>=4.2.0.32
* matplotlib==3.4.1
* argparse==1.4.0
* scipy==1.6.2
* scikit-learn==0.24.1
* pillow>=8.3.2
* streamlit==0.79.0
Ces différents prérequis se trouvent dans le fichier [requirements.txt](https://github.com/BasileAmeeuw/IA-project-orientation-and-mood-detection/blob/main/requirements.txt)

Ensuite il vous faudra également installer dlib qui est une librairie C++ mais utilisable en Python, pour l'installer suivez les instructions détaillées dans le README suivant: [README Dlib](https://github.com/BasileAmeeuw/IA-project-orientation-and-mood-detection/blob/main/dlib-19.9/README.md)

## Exécution pour l'exécution

#### Avec modèle déja entrainé

Si vous utilisez les modèles pré-entrainés: il vous suffit d'exécuter la commande suivante
```
python main.py
```

Si vous voulez utiliser une vidéo pré-enregistrer exécutez ceci
```
python main.py --video PATH_OF_VIDEO
```

Et si vous souhaiter utiliser une image exécutez ceci:
```
python main.py --image PATH_OF_IMAGE
```

#### Avec entrainement

Si vous voulez entrainer les modèles vous même, vous devrez dans un premier temps entrainer le modèle pour le mask et ensuite celui pour les émotions. Dans un premier temps exécutez:
```
python mask_training.py
```
Dans un deuxième temps exécutes:
```
python emotion_training.py
```
 
 **Remarque:** Il est possible d'effectuer ces entrainements avec certains arguments et pour cela je vous suggère de rentrer dans les fichier Python.
 
 Après les entrainements retourner à l'étape "avec modèle déja entrainé" et suivez les instructions.
 
 ## Conclusion
 
 Le projet est suffisant selon nous mais exige encore beaucoup de révision et de versioning pour arriver à une application fiable et viable.
 
 ## Contributeurs
 
 [Leprêtre Romain](https://github.com/rlepretre)
 
 [Ameeuw Basile](https://github.com/BasileAmeeuw)
 
 ## Licence 
 
 MIT License
