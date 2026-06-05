import pandas as pd
import numpy as np
from meditation.params import *

def path_data(sujets=list, labels=list, sessions=list, root=ROOT) -> dict:

    """Pour une liste de sujets (int) donnés, et la phase et session correspondante,
    renvoie un dictionnaire ayant pour clé ces informations et pour valeur le PATH
    de la séquence donnée"""

    sujets_str = []
    for sujet in sujets:
        if len(str(sujet))==1:
            sujets_str.append('00'+str(sujet))
        else:
            sujets_str.append('0'+str(sujet))

    dict = {f'{sujets_str[0]}_{labels[0]}_{sessions[0]}': f'{root}/raw_data/derivatives/ml_preproc_data/sub-{sujets_str[0]}/sub-{sujets_str[0]}_ses-{sessions[0]}_task-{labels[0]}_eeg_preproc.npy'}
    for sujet in sujets_str:
        for label in labels:
            for session in sessions:
                dict[f'{sujet}_{label}_{session}'] = f'{root}/raw_data/derivatives/ml_preproc_data/sub-{sujet}/sub-{sujet}_ses-{session}_task-{label}_eeg_preproc.npy'
    return dict

def slice_data(data, window_size=1000, step=1000, start=0) -> list:

        """Renvoie une liste de la données séparée (sur les lignes) de chaque window_size.
        """

        data_sliced = []
        index = start
        while index + window_size <= np.shape(data)[0]:
            data_sliced.append(data[index : index + window_size,:])
            index = index + step
        return data_sliced

def load_data_dict(paths=dict, window_size=1000, step=1000, start=0):

    """"
    - renvoie un tuple contenant X de shape (nb de seq, window_size, nb de canaux)
    et y de shape (nb de seq,) à partir d'un dictionnaire de path.
    - start corresponds au début de la 1ère séquence

    """

    def is_medit(key=str):

        """renvoie 0 (repos) ou 1 (méditation) selon le label"""

        label = key.split(sep='_')[1]
        if label in ('restCE01','restCE02', 'restOE'):
            return 0
        elif label in ('Medita', 'slMedita'):
            return 1
        else:
            print(f'label {label} does not exist')
            return None


    X_final = []
    y_final = []

    for key, path in paths.items():
        data = np.load(path).T
        X = np.array(slice_data(data, window_size=window_size, step=step, start=start))
        y = np.array([is_medit(key) for i in range(len(X))])
        X_final.extend(X)
        y_final.extend(y)

    return np.array(X_final), np.array(y_final)

def load_data(sujets=None,
              labels=['Medita', 'slMedita', 'restCE01', 'restCE02', 'restOE'],
              sessions=['premedita'],
              window_size=1000,
              step=1000,
              start=0,
              split=None,
              root=ROOT) -> tuple:

    """"
    - renvoie un tuple contenant :
        - X de shape (nb de seq, window_size, nb de canaux=64) contenant la donnée
          séparée de window_size
        - y de shape (nb de seq,) contenant le label 0 ou 1 (repos ou méditation)
    - sujets : liste de int entre 1 et 74
    - labels : liste de str parmi {'Medita', 'slMedita', 'restCE01', 'restCE02', 'restOE'}
    - sessions : liste de str parmi {'premedita', 'posmedita'}
    - start correspond au début de la 1ère séquence, par défaut à 0
    - window_size correspond à la largueur d'une sequence, 1000 points (4s) par défaut
    - step correspond au décalage de chaque window
    - split permet d'assigner automatiquement des sujets (si sujets=None),
      parmi {'train', 'val', 'test', 'train_small', 'val_small', 'test_small'}

    """

    if sujets == None:
        if split == 'train':
            sujets = TRAIN
        elif split == 'val':
            sujets = VAL
        elif split == 'test':
            sujets = TEST
        elif split == 'train_small':
            sujets = TRAIN_SMALL
        elif split == 'val_small':
            sujets = VAL_SMALL
        elif split == 'test_small':
            sujets = TEST_SMALL
        else:
            print('no subject selected')
            return None

    return load_data_dict(path_data(sujets=sujets, labels=labels, sessions=sessions, root=root),
                          window_size=window_size, step=step,
                          start=start)
