import requests
import json

url_regresion = "http://localhost:8000/regresion"
url_svm = "http://localhost:8000/svm"
url_tree = "http://localhost:8000/tree"
url_knn = "http://localhost:8000/knn"

datos_ejemplo_1 = {"longitud_petalo": 1.5, "ancho_petalo": 0.3}
datos_ejemplo_2 = {"longitud_petalo": 4.0, "ancho_petalo": 1.0}

def realizar_peticiones(url, datos):
    headers = {'Content-Type': 'application/json'}
    for i, dato in enumerate(datos, start=1):
        response = requests.post(url, data=json.dumps(dato), headers=headers)
        if response.status_code == 200:
            print(f"Respuesta {i} del servicio {url}: {response.text}")
        else:
            print(f"Error en la petición {i} al servicio {url}. Código de estado: {response.status_code}")

realizar_peticiones(url_regresion, [datos_ejemplo_1, datos_ejemplo_2])
realizar_peticiones(url_svm, [datos_ejemplo_1, datos_ejemplo_2])
realizar_peticiones(url_tree, [datos_ejemplo_1, datos_ejemplo_2])
realizar_peticiones(url_knn, [datos_ejemplo_1, datos_ejemplo_2])
