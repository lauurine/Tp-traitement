import os
from typing import List, Tuple
import numpy as np
from sklearn.metrics import roc_curve, auc
from scipy.ndimage import convolve
import cv2
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread


############################################################################
#                          MSLD IMPLEMENTATION                             #
############################################################################


# Pour chaque longueur de ligne l dans la liste L, un masque initial (line_detector) est créé avec une ligne horizontale au centre.
# Ce masque initial est ensuite rotatif pour créer des masques pour différentes orientations. La rotation est effectuée en utilisant cv2.getRotationMatrix2D et cv2.warpAffine.
# Tous les masques résultants sont empilés le long de l'axe 2 pour former un masque 3D qui est ensuite stocké dans self.line_detectors_masks.
class MSLD:
    """
    Classe implémentant l'algorithme de MSLD, ainsi que différents outils de
    mesure de performances.

    Les attributs de cette classe sont:
        W: Taille de la fenêtre.
        L: Vecteur contenant les longueurs des lignes à détecter.
        n_orientation: Nombre d'orientation des lignes à détecter.
        threshold: Seuil de segmentation (à apprendre).
        line_detectors_masks: Masques pour la détection des lignes pour chaque valeur de L et chaque valeur de
            n_orientation.
        avg_mask: Masque moyenneur de taille W x W.
    """

    def __init__(self, W: int, L: List[int], n_orientation: int) -> None:
        """Constructeur qui initialise un objet de type MSLD. Cette méthode est appelée par
        >>> msld = MSLD(W=..., L=..., n_orientation=...)

        Args:
            W (int): Taille de la fenêtre (telle que définie dans l'article).
            L (List[int]): Une liste contenant les valeurs des longueurs des lignes qui seront détectées par la MSLD.
            n_orientation (int): Nombre d'orientations des lignes à détecter.
        """
        self.W = W
        self.L = L
        self.n_orientation = n_orientation

        self.threshold = 0.5  # On choisit un seuil par défaut de 0.5, mais le seuil optimal sera appris par la suite.

        # TODO: 1.2.Q1
        self.avg_mask = np.ones((W, W)) * 1/(W**2)


        # TODO: 1.2.Q2
        # line_detectors_masks est un dictionnaire contenant les masques
        # de détection de ligne pour toutes les échelles contenues
        # dans la liste L et pour un nombre d'orientation égal à
        # n_orientation. Ainsi pour toutes les valeurs de L:
        # self.line_detectors_masks[l] est une matrice de la forme [l,l,n_orientation]

        self.line_detectors_masks = {}
        for l in L:
            # On calcule le détecteur de ligne initial de taille l (les dimensions du masque sont lxl).
            line_detector = np.zeros((l,l), dtype=np.float32) 
            ligne_centrale = l // 2
            line_detector[ligne_centrale, :] = 1/l
            
            # On initialise la liste des n_orientation masques de taille lxl.
            line_detectors_masks = []
            # On effectue n_orientation-1 rotations du masque line_detector.
            # Pour un angle donné, la rotation sera effectuée par
         
            #vecteur = np.arange(0,181,180 / n_orientation)
            for angle in range(0,180, int(180/self.n_orientation)):
                r = cv2.getRotationMatrix2D((l // 2, l // 2), angle, 1)
                rotated_mask = cv2.warpAffine(line_detector, r, (l, l))
                line_detectors_masks.append(rotated_mask / rotated_mask.sum())

            # On assemble les n_orientation masques ensemble:
            self.line_detectors_masks[l] = np.stack(line_detectors_masks, axis=2)
            #nous avons une matrice 3D avec a chaque tranche une orientation differente 

    def basic_line_detector(self, grey_lvl: np.ndarray, L: int) -> np.ndarray:
        """Applique l'algorithme Basic Line Detector sur la carte d'intensité grey_lvl avec des lignes de longueurs L.

        Args:
            grey_lvl (np.ndarray): Carte d'intensité 2D avec dtype float sur laquelle est appliqué le BLD.
            L (int): Longueur des lignes (on supposera que L est présent dans self.L et donc que
                self.line_detectors_masks[L] existe).

        Returns:
            R (np.ndarray): Carte de réponse 2D en float du Basic Line Detector.
        """
        
        # TODO: 1.2.Q3
        # Les masques de détections de lignes de longueur L initialisés dans le constructeur sont accessibles par:
        # self.line_detectors_masks[L]
        I_avg = convolve(grey_lvl, self.avg_mask)

        line_detector = self.line_detectors_masks[L]

        matrice_max = np.zeros(grey_lvl.shape)
        for n in range (0, (self.n_orientation) ):
            convolution = convolve( grey_lvl, line_detector[:,:,n])
            matrice_max = np.maximum(convolution, matrice_max)


        R = matrice_max - I_avg

        R_prime = (R - np.mean(R)) / np.std(R)

        return R_prime

    def multi_scale_line_detector(self, image: np.ndarray) -> np.ndarray:
        """Applique l'algorithme de Multi-Scale Line Detector et combine les réponses des BLD pour obtenir la carte
        d'intensité de l'équation 4 de la section 3.3 Combination Method.

        Args:
            image (np.ndarray): Image RGB aux intensitées en float comprises entre 0 et 1 et de dimensions
                (hauteur, largeur, canal) (canal: R=1 G=2 B=3)

        Returns:
            Rcombined (np.ndarray): Carte d'intensité combinée.
        """

        # TODO: 1.3.Q1
        # Pour les hyperparamètres L et W utilisez les valeurs de self.L et self.W.
        #rappel : ici on parle de image mais il faudra entrer le canal vert inversé dans l'autre fichier pour appeler cette fonction
        coefficient = (1 / (self.n_orientation +1 )) 
        Rcombined = np.zeros(image.shape)
        for i in range(0, len(self.L)):
            reponse = self.basic_line_detector( image, (self.L)[i])
            Rcombined += reponse
        Rcombined = coefficient * (Rcombined + image)

        return Rcombined

    def learn_threshold(self, dataset: List[dict]) -> Tuple[float, float]:
        """
        Apprend le seuil optimal pour obtenir la précision la plus élevée
        sur le dataset donné.
        Cette méthode modifie la valeur de self.threshold par le seuil
        optimal puis renvoie ce seuil et la précision obtenue.

        Args:
            dataset (List[dict]): Liste de dictionnaires contenant les champs ["image", "label", "mask"].

        Returns:
            threshold (float): Seuil proposant la meilleure précision
            accuracy (float): Valeur de la meilleure précision
        """

        fpr, tpr, thresholds = self.roc(dataset)
        # # TODO: 1.4.Q3
        # # Utilisez np.argmax pour trouver l'indice du maximum d'un vecteur.
        # y_true = []
        # y_pred = []

        # for d in dataset:
        #     # Pour chaque élément de dataset
        #     label = d["label"]  # On lit le label
        #     mask = d["mask"]  # le masque
        #     image = d["image"]  # et l'image de l'élément.

        #     # On calcule la prédiction du msld sur cette image.
        #     prediction = self.multi_scale_line_detector(image)

        #     # On applique les masques à label et prediction pour qu'ils contiennent uniquement
        #     # la liste des pixels qui appartiennent au masque.
        #     label = label[mask]
        #     prediction = prediction[mask]

        #     # On ajoute les vecteurs label et prediction aux listes y_true et y_pred
        #     y_true.append(label)
        #     y_pred.append(prediction)

        # # On concatène les vecteurs de la listes y_true pour obtenir un unique vecteur contenant
        # # les labels associés à tous les pixels qui appartiennent au masque du dataset.
        # y_true = np.concatenate(y_true)
        # # Même chose pour y_pred.
        # y_pred = np.concatenate(y_pred)
        
        
        accuracies = [(tpr_i * sum(self.y_true) + (1 - fpr_i) * (len(self.y_true) - sum(self.y_true))) / len(self.y_true) for tpr_i, fpr_i in zip(tpr, fpr)]

        # #accuracies = [fpr_i/ tpr_i for tpr_i, fpr_i in zip(tpr, fpr)]
        # # Trouver l'indice du seuil qui maximise l'accuracy
        best_threshold_index = np.argmax(accuracies)
        
        threshold = thresholds[best_threshold_index]
        accuracy = accuracies[best_threshold_index]

        self.threshold = threshold

        # print("Seuil optimal pour la meilleure accuracy :", self.threshold)
        # print("Meilleure accuracy :", accuracy)
        return threshold, accuracy
        #return fpr, tpr, thresholds

    def segment_vessels(self, image: np.ndarray) -> np.ndarray:
        """
        Segmente les vaisseaux sur une image en utilisant la MSLD.

        Args:
            image (np.ndarray): Image RGB sur laquelle appliquer l'algorithme MSLD.

        Returns:
            vessels (np.ndarray): Carte binaire 2D de la segmentation des vaisseaux.
        """

        # TODO: 1.5.Q1
        # Utilisez self.multi_scale_line_detector(image) et self.threshold.

        vessels = ...

        return vessels

    ############################################################################
    #                           Visualisation                                  #
    ############################################################################
    def show_diff(self, sample: dict) -> None:
        """Affiche la comparaison entre la prédiction de l'algorithme et les valeurs attendues (labels) selon le code
        couleur suivant:
           - Noir: le pixel est absent de la prédiction et du label
           - Rouge: le pixel n'est présent que dans la prédiction
           - Bleu: le pixel n'est présent que dans le label
           - Blanc: le pixel est présent dans la prédiction ET le label

        Args:
            sample (dict): Un échantillon provenant d'un dataset contenant les champs ["data", "label", "mask"].
        """
        # Calcule la segmentation des vaisseaux
        pred = self.segment_vessels(sample["image"])

        # Applique le masque à la prédiction et au label
        pred = pred & sample["mask"]
        label = sample["label"] & sample["mask"]

        # Calcule chaque canal de l'image:
        # rouge: 1 seulement pred est vrai, 0 sinon
        # bleu: 1 seulement si label est vrai, 0 sinon
        # vert: 1 seulement si label et pred sont vrais (de sorte que la couleur globale soit blanche), 0 sinon
        red = pred * 1.0
        blue = label * 1.0
        green = (pred & label) * 1.0

        rgb = np.stack([red, green, blue], axis=2)
        plt.imshow(rgb)
        plt.axis("off")
        plt.title("Différences entre la segmentation prédite et attendue")

    ############################################################################
    #                         Segmentation Metrics                             #
    ############################################################################
    def roc(self, dataset: List[dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calcule la courbe ROC de l'algorithme MSLD sur un dataset donné et sur la région d'intérêt indiquée par le
        champ "mask".

        Args:
            dataset (List[dict]): Base de données sur laquelle calculer la courbe ROC.

        Returns:
            fpr (np.ndarray): Vecteur float des taux de faux positifs.
            tpr (np.ndarray): Vecteur float des taux de vrais positifs.
            thresholds (np.ndarray): Vecteur float des seuils associés à ces taux.
        """

        y_true = []
        y_pred = []

        for d in dataset:
            # Pour chaque élément de dataset
            label = d["label"]  # On lit le label
            mask = d["mask"]  # le masque
            image = d["image"]  # et l'image de l'élément.
            # image= 1 - image[:,:,1]
            # On calcule la prédiction du msld sur cette image.
            prediction = self.multi_scale_line_detector(image)

            # On applique les masques à label et prediction pour qu'ils contiennent uniquement
            # la liste des pixels qui appartiennent au masque.
            label = label[mask]
            prediction = prediction[mask]

            # On ajoute les vecteurs label et prediction aux listes y_true et y_pred
            y_true.append(label)
            y_pred.append(prediction)

        # On concatène les vecteurs de la listes y_true pour obtenir un unique vecteur contenant
        # les labels associés à tous les pixels qui appartiennent au masque du dataset.
        y_true = np.concatenate(y_true)
        # Même chose pour y_pred.
        y_pred = np.concatenate(y_pred)

        self.y_true = y_true
        
        # On calcule le taux de vrai positif et de faux positif du dataset pour chaque seuil possible.
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        return fpr, tpr, thresholds

    def naive_metrics(self, dataset: List[dict]) -> Tuple[float, np.ndarray]:
        """
        Évalue la précision et la matrice de confusion de l'algorithme sur
        un dataset donné et sur la région d'intérêt indiquée par le
        champs mask.

        Args:
            dataset (List[dict]): Base de données sur laquelle calculer les métriques.

        Returns:
            accuracy (float): Précision.
            confusion_matrix (np.ndarray): Matrice de confusion 2 x 2 normalisée par le nombre de labels positifs et
                négatifs.
        """

        # TODO: 2.1.Q1

        accuracy = ...
        confusion_matrix = ...

        return accuracy, confusion_matrix

    def dice(self, dataset: List[dict]) -> float:
        """
        Évalue l'indice Sørensen-Dice de l'algorithme sur un dataset donné et sur la région d'intérêt indiquée par le
        champ "mask".

        Args:
            dataset (List[dict]): Base de données sur laquelle calculer l'indice Dice.

        Returns:
            dice_index (float): Indice de Sørensen-Dice.
        """

        # TODO: 2.2.Q2
        # Vous pouvez utiliser la fonction fournie tout en bas de ce fichier: dice().

        dice_index = ...

        return dice_index

    def plot_roc(self, dataset: List[dict]) -> float:
        """
        Affiche la courbe ROC et calcule l'AUR de l'algorithme pour un
        dataset donnée et sur la région d'intérêt indiquée par le champs
        mask.

        Args:
            dataset (List[dict]): Base de données sur laquelle calculer l'AUR.

        Returns:
            roc_auc (float): Aire sous la courbe ROC.
        """

        # TODO: 2.3.Q2
        # Utilisez la méthode self.roc(dataset) déjà implémentée.

        roc_auc = ...

        return roc_auc


def load_dataset() -> Tuple[List[dict], List[dict]]:
    """Charge les images des ensembles d'entrainement et de test dans 2 listes de dictionnaires. Pour chaque
    échantillon, il faut créer un dictionnaire dans le dataset contenant les champs ["name", "image", "label", "mask"].
    On pourra ainsi accéder à la première image du dataset d'entrainement avec train[0]["image"].

    Returns:
        train (List[dict]): Liste de dictionnaires contenant les champs ["name", "image", "label", "mask"] pour les
            images d'entrainement.
        test (List[dict]): Liste de dictionnaires contenant les champs ["name", "image", "label", "mask"] pour les
            images de test.
    """
    files = sorted(os.listdir(r"DRIVE/data/training"))
    train = []

    for file in files :
        sample = {}
        sample["name"] =file

        image = imread(os.path.join(r"DRIVE/data/training", file))
        sample["image"] = (image).astype(float)
        sample["label"] = imread(os.path.join(r"DRIVE/label/training", file)).astype(bool)
        sample["mask"] = imread(os.path.join(r"DRIVE/mask/training", file)).astype(bool)
        train.append(sample)

    files = sorted(os.listdir(r"DRIVE/data/test"))
    test = []

    for file in files :
        sample = {}
        sample["name"] =file

        image = imread(os.path.join(r"DRIVE/data/test", file))
        sample["image"] = imread(os.path.join(r"DRIVE/data/test", file)).astype(float)
        sample["label"] = imread(os.path.join(r"DRIVE/label/test", file)).astype(bool)
        sample["mask"] = imread(os.path.join(r"DRIVE/mask/test", file)).astype(bool)
        test.append(sample)

    return train, test



def dice(targets: np.ndarray, predictions: np.ndarray) -> float:
    """Calcule l'indice de Sørensen-Dice entre les prédictions et la vraie segmentation. Les deux arrays doivent avoir
    la même forme.

    Args:
        targets (np.ndarray): Vraie segmentation.
        predictions (np.ndarray): Prédiction de la segmentation.

    Returns:
        dice_index (float): Indice de Sørensen-Dice.
    """
    
        

    dice_index = 2 * np.sum(targets * predictions) / (targets.sum() + predictions.sum())
    return dice_index
