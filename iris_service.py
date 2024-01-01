import numpy as np
from sklearn import datasets
import pickle

def predict_single(datos, model):
    iris = datasets.load_iris()
    with open('models/scaler.pkl', 'rb') as scaler_file:
        scaler_loaded = pickle.load(scaler_file)

    nuevos_datos = np.array([datos["longitud_petalo"], datos["ancho_petalo"]]).reshape(1, -1)
    nuevos_datos_std = scaler_loaded.transform(nuevos_datos)

    prediccion = model.predict(nuevos_datos_std)
    return iris.target_names[prediccion[0]]

def getRegressionModel ():
    with open('models/model-logistic-regression.pck', 'rb') as regresion_file:
        regresion = pickle.load(regresion_file)
    return regresion

def getSVMModel ():
    with open('models/model-svm.pck', 'rb') as svm_file:
        svm = pickle.load(svm_file)
    return svm

def getTreeModel ():
    with open('models/model-tree.pck', 'rb') as tree_file:
        tree = pickle.load(tree_file)
    return tree

def getKNNModel ():
    with open('models/model-logistic-regression.pck', 'rb') as knn_file:
        knn = pickle.load(knn_file)
    return knn