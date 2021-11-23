from sklearn.model_selection import train_test_split
import numpy as np

import pickle
import tarfile
import os


def reshape_cifar(X_data):
    """Cambia las dimensiones de los objetos guardados en CIFAR10.

    De forma nativa, cada imagen se representa por un arreglo de largo 3072. Para representar
    esta información como una imagen, se cambia la dimensión de cada arreglo a 32x32x3.


    Parameters
    ----------
    X_data : numpy Array
        Array de numpy de largo 3072.
    
    Returns
    -------
    X_reshaped : numpy Array
        Array de numpy con dimensiones (n_img, 32, 32, 3).
    """
    X_reshaped = X_data.reshape([-1, 3, 32, 32])
    X_reshaped = X_reshaped.transpose([0, 2, 3, 1])
    return X_reshaped

def get_cifar10():
    """Preprocesamiento del dataset CIFAR10.

    El dataset se encuentra almacenado en un archivo tar, del cuál se extraen 5 batches con datos para el
    entrenamiento y un batch con datos para el test, así como un archivo que contiene el mapeo entre
    el número de cada etiqueta y su nombre.
    
    Returns
    -------
    X_train, y_train, X_test, y_test : numpy Array
        Ejemplos y etiquetas separadas para entrenamiento, validación y test.
    
    label_names : list
        lista que en cada índice contiene el nombre de la etiqueta correspondiente.
    """
    is_available = os.path.isfile("cifar-10-python.tar.gz")
    if not is_available:
        raise FileNotFoundError("CIFAR dataset not found.")

    label_names = []

    X_train = []
    y_train = []
    X_test = []
    y_test = []
    
    with tarfile.open("cifar-10-python.tar.gz", mode="r:gz") as tar_cifar:
        with tar_cifar.extractfile("cifar-10-batches-py/batches.meta") as meta:
            meta_data = pickle.load(meta, encoding="latin1")
            label_names = meta_data["label_names"]

        for i in range(1, 6):
            with tar_cifar.extractfile(f"cifar-10-batches-py/data_batch_{i}") as batch:
                train_batch = pickle.load(batch, encoding="latin1")
                X_train.append(train_batch["data"])
                y_train += train_batch["labels"]
        X_train = np.concatenate(X_train, axis=0)
        y_train = np.array(y_train)

        with tar_cifar.extractfile("cifar-10-batches-py/test_batch") as batch:
            test_batch = pickle.load(batch, encoding="latin1")
            X_test = test_batch["data"]
            y_test = np.array(test_batch["labels"])

    X_train = reshape_cifar(X_train)
    X_test = reshape_cifar(X_test)

    return X_train, y_train, X_test, y_test, label_names

def get_validation_data(x, y, val_size, seed=None):
    """Particiona el conjunto de entrenamiento en un conjunto de entrenamiento y otro de validación.

    Parameters
    ----------
    x : numpy Array
        Datos de entrenamiento.
    y : numpy array
        Etiquetas de entrenamiento.
    val_size : int o float
        Cantidad de datos destinados al conjunto de validación. De ser un entero, debe ser menor a la
        cantidad de datos en x; de ser un float, debe estar entre 0 y 1.
    seed : int
        Semilla utilizada para fijar la secuencia pseudo aleatorio con motivos de reproductibilidad.
    
    Returns
    -------
    X_train, y_train, X_val, y_val : numpy Array
        Ejemplos y etiquetas separadas para entrenamiento y validación.
    """
    X_train, X_val, y_train, y_val = train_test_split(
        x, y, test_size=val_size, stratify=y, random_state=seed)

    return X_train, y_train, X_val, y_val

def get_cifar10_data(val_size, seed=None):
    """Dataset CIFAR10 con nombre de etiquetas separado en entrenamiento, validación y test.

    Parameters
    ----------
    val_size : int o float
        Cantidad de datos destinados al conjunto de validación. De ser un entero, debe ser menor a la
        cantidad de datos en x; de ser un float, debe estar entre 0 y 1.
    seed : int
        Semilla utilizada para fijar la secuencia pseudo aleatorio con motivos de reproductibilidad.
    
    Returns
    -------
    X_train, y_train, X_val, y_val, X_test, y_test : numpy Array
        Ejemplos y etiquetas separadas para entrenamiento, validación y test.
    
    label_names : list
        lista que en cada índice contiene el nombre de la etiqueta correspondiente.
    """
    X_train, y_train, X_test, y_test, label_names = get_cifar10()
    X_train, y_train, X_val, y_val = get_validation_data(X_train, y_train, val_size, seed=seed)

    X_train = (X_train/255).astype(np.float32)
    X_val = (X_val/255).astype(np.float32)
    X_test = (X_test/255).astype(np.float32)

    return X_train, y_train, X_val, y_val, X_test, y_test, label_names