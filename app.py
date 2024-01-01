from flask import Flask, request
from iris_service import predict_single, getRegressionModel, getSVMModel, getTreeModel, getKNNModel

app = Flask('predict')

@app.route('/regresion', methods=['POST'])
def predict_regression():
    datos = request.get_json()
    regresion = getRegressionModel()
    prediction = predict_single(datos, regresion)
    
    return prediction


@app.route('/svm', methods=['POST'])
def predict_svm():
    datos = request.get_json()
    svm = getSVMModel()
    prediction = predict_single(datos, svm)
    
    return prediction


@app.route('/tree', methods=['POST'])
def predict_tree():
    datos = request.get_json()
    tree = getTreeModel()
    prediction = predict_single(datos, tree)
    
    return prediction


@app.route('/knn', methods=['POST'])
def predict_knn():
    datos = request.get_json()
    knn = getKNNModel()
    prediction = predict_single(datos, knn)
    
    return prediction


if __name__ == '__main__':
    app.run(debug=True, port=8000)  