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


def load_data_dict(paths=dict, window_size=1000, step=1000, start=0):

    """"
    - renvoie un tuple contenant X de shape (nb de seq, window_size, nb de canaux)
    et y de shape (nb de seq,) à partir d'un dictionnaire de path.
    - start corresponds au début de la 1ère séquence

    """

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

    """
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
    - root le chemin contenant le dossier 'raw'

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

def load_one_subject(sujet=int,
                     labels=['Medita', 'slMedita', 'restCE01', 'restCE02', 'restOE'],
                     sessions=['premedita'],
                     window_size=1000,
                     step=1000,
                     start=0,
                     root=ROOT,
                     train_split=0.8,
                     gap=0
                     ):

    """
    - renvoie un tuple contenant (X_train, X_test, y_train, y_test):
        - X de shape (nb de seq, window_size, nb de canaux=64) contenant la donnée
          séparée de window_size
        - y de shape (nb de seq,) contenant le label 0 ou 1 (repos ou méditation)
    - sujets : liste de int entre 1 et 74
    - labels : liste de str parmi {'Medita', 'slMedita', 'restCE01', 'restCE02', 'restOE'}
    - sessions : liste de str parmi {'premedita', 'posmedita'}
    - start correspond au début de la 1ère séquence, par défaut à 0
    - window_size correspond à la largueur d'une sequence, 1000 points (4s) par défaut
    - step correspond au décalage de chaque window
    - root le chemin contenant le dossier 'raw'
    - train_split est le ratio de données train (entre 0 et 1)
    - gap est le nombre de données ignorées entre le train et le test

    """

    # Get data path
    paths = path_data([sujet], labels, sessions, root)

    X_train_final = []
    X_test_final = []
    y_train_final = []
    y_test_final = []

    for key, path in paths.items():
        # Load data
        data = np.load(path).T

        # Split data (temporally) between train and test, with gap between train and test
        data_train = data[:int(train_split*data.shape[0]),:]
        data_test = data[int(train_split*data.shape[0]) + gap:,:]

        # Apply windowing to build X, y
        X_train = np.array(slice_data(data_train, window_size=window_size, step=step, start=start))
        y_train = np.array([is_medit(key) for i in range(len(X_train))])
        X_test = np.array(slice_data(data_test, window_size=window_size, step=step, start=0))
        y_test = np.array([is_medit(key) for i in range(len(X_test))])

        # Stack the resulting windows
        X_train_final.extend(X_train)
        y_train_final.extend(y_train)
        X_test_final.extend(X_test)
        y_test_final.extend(y_test)

    return np.array(X_train_final), np.array(X_test_final), np.array(y_train_final), np.array(y_test_final)




def load_train_val_test_one_subject(sujet=int,
                                    labels=['Medita', 'slMedita', 'restCE01', 'restCE02', 'restOE'],
                                    sessions=['premedita'],
                                    window_size=1000,
                                    step=1000,
                                    start=0,
                                    root=ROOT,
                                    train_split=0.8,
                                    val_split=0.2,
                                    gap_train_val=0,
                                    gap_train_test=0
                                    ):

    """
    - renvoie un tuple contenant (X_train, X_val, X_test, y_train, y_val, y_test):
        - X de shape (nb de seq, window_size, nb de canaux=64) contenant la donnée
          séparée de window_size
        - y de shape (nb de seq,) contenant le label 0 ou 1 (repos ou méditation)
    - sujets : liste de int entre 1 et 74
    - labels : liste de str parmi {'Medita', 'slMedita', 'restCE01', 'restCE02', 'restOE'}
    - sessions : liste de str parmi {'premedita', 'posmedita'}
    - start correspond au début de la 1ère séquence, par défaut à 0
    - window_size correspond à la largueur d'une sequence, 1000 points (4s) par défaut
    - step correspond au décalage de chaque window
    - root le chemin contenant le dossier 'raw'
    - train_split est le ratio de données train (entre 0 et 1)
    - val_split est le ratio de données de validation par rapport aux données train (entre 0 et 1)
    - gap_train_val est le nombre de données ignorées entre le train et le val
    - gap_train_test est le nombre de données ignorées entre le train et le test

    """
    # Get data path
    paths = path_data([sujet], labels, sessions, root)

    X_train_final = []
    X_val_final = []
    X_test_final = []
    y_train_final = []
    y_val_final = []
    y_test_final = []

    for key, path in paths.items():
        # Load data
        data = np.load(path).T

        # Get train and val slice
        train_slice = int(train_split*data.shape[0])
        val_slice = int(train_slice*val_split)

        # Split data (temporally) between train, val and test, with gap between train, val and test
        data_train = data[: train_slice - val_slice,:]
        data_val = data[train_slice - val_slice + gap_train_val: train_slice,:]
        data_test = data[train_slice + gap_train_test:,:]

        # Apply windowing to build X, y
        X_train = np.array(slice_data(data_train, window_size=window_size, step=step, start=start))
        y_train = np.array([is_medit(key) for i in range(len(X_train))])
        X_val = np.array(slice_data(data_val, window_size=window_size, step=step, start=0))
        y_val = np.array([is_medit(key) for i in range(len(X_val))])
        X_test = np.array(slice_data(data_test, window_size=window_size, step=step, start=0))
        y_test = np.array([is_medit(key) for i in range(len(X_test))])

        # Stack the resulting windows
        X_train_final.extend(X_train)
        y_train_final.extend(y_train)
        X_val_final.extend(X_val)
        y_val_final.extend(y_val)
        X_test_final.extend(X_test)
        y_test_final.extend(y_test)

    return np.array(X_train_final), np.array(X_val_final), np.array(X_test_final), np.array(y_train_final), np.array(y_val_final), np.array(y_test_final)



def which_medit(key, data):
    """
    renvoie 0 (HK), 1 (SA) ou 2 (BF) selon le label.

    - key est la clé du dictionnaire créé par path_data
    - data est un DataFrame contenant participant_id de valeur sub-xx ainsi que group correspondant au label

    """

    sujet_str = key.split(sep='_')[0][1:]

    try:
        group_name = data.loc[data['participant_id']== f'sub-{sujet_str}', 'group'].iloc[0]
    except IndexError:
        group_name = None

    if group_name == 'HK':
        return 0
    if group_name == 'BF':
        return 1
    if group_name == 'SA':
        return 2
    else:
        print(f'subject {sujet_str} does not exist or does not have post-meditation data')
        return None



def load_data_task_2(sujets=list, labels=['Medita','slMedita'], window_size=1000, step=1000, start=0, root=ROOT):

    """
    Load la data de la session post-meditation
    - renvoie un tuple contenant :
        - X de shape (nb de seq, window_size, nb de canaux=64) contenant la donnée
          séparée de window_size
        - y de shape (nb de seq,) contenant le label 0 (HK), 1 (BF), ou 2 (SA)
    - sujets : liste de int entre 1 et 74
    - labels : liste de str parmi {'Medita', 'slMedita'}
    - start correspond au début de la 1ère séquence, par défaut à 0
    - window_size correspond à la largueur d'une sequence, 1000 points (4s) par défaut
    - step correspond au décalage de chaque window
    - root le chemin contenant le dossier 'raw'

    """

    # Get data paths with posmedita according to task 2
    paths = path_data(sujets=sujets, labels=labels, sessions=['posmedita'], root=root)

    X_final = []
    y_final = []

    # Get labels from participants.tsv
    df = pd.read_csv(f'{root}/raw_data/participants.tsv',sep='\t')
    df = df.dropna(subset=['post_test'])

    # Append X, y for each selected path
    for key, path in paths.items():
        sujet_str = key.split(sep='_')[0][1:]
        try:
            data = np.load(path).T
        except FileNotFoundError:
            print(f'subject {sujet_str} does not exist or does not have post-meditation data')
            continue
        X = np.array(slice_data(data, window_size=window_size, step=step, start=start))
        y = np.array([which_medit(key, df) for i in range(len(X))])
        X_final.extend(X)
        y_final.extend(y)
    return np.array(X_final), np.array(y_final)
