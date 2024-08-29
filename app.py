import logging
from flask import Flask, request, jsonify
import cv2
import numpy as np
import mediapipe as mp
import tempfile
import os

app = Flask(__name__)

# Configuración del logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Inicializar Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

def detect_dark_areas(region):
    gray_region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    alpha = 1.2  # Menor factor de contraste
    beta = -20  # Menor ajuste de brillo
    adjusted_region = cv2.convertScaleAbs(gray_region, alpha=alpha, beta=beta)
    blurred_region = cv2.GaussianBlur(adjusted_region, (5, 5), 0)
    _, thresh = cv2.threshold(blurred_region, 80, 255, cv2.THRESH_BINARY_INV)  # Ajuste del umbral
    dark_areas = cv2.countNonZero(thresh)
    total_area = region.shape[0] * region.shape[1]
    percentage_oje = (dark_areas / total_area) * 100
    return percentage_oje

def detect_wrinkles(region):
    gray_region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    alpha = 1.5  # Menor factor de contraste
    beta = -20  # Menor ajuste de brillo
    adjusted_region = cv2.convertScaleAbs(gray_region, alpha=alpha, beta=beta)
    blurred_region = cv2.GaussianBlur(adjusted_region, (3, 3), 0)  # Menor suavizado
    edges = cv2.Canny(blurred_region, 50, 150)  # Ajustar bordes
    _, thresh = cv2.threshold(edges, 80, 255, cv2.THRESH_BINARY)  # Ajuste del umbral
    wrinkles = cv2.countNonZero(thresh)
    total_area = region.shape[0] * region.shape[1]
    percentage_arr = (wrinkles / total_area) * 100
    return percentage_arr

@app.route('/predict', methods=['POST'])
def predict():
    try:
        logger.info("Recibiendo la solicitud de predicción")
        file = request.files.get('file')
        if not file:
            logger.error("No se ha proporcionado ningún archivo")
            return jsonify({"error": "No se ha proporcionado ningún archivo"}), 400

        logger.info("Guardando archivo temporalmente")
        # Guardar el archivo temporalmente
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            temp_file.write(file.read())
            temp_path = temp_file.name

        logger.info("Cargando imagen con OpenCV")
        # Cargar la imagen con OpenCV
        image = cv2.imread(temp_path)
        if image is None:
            logger.error("No se pudo cargar la imagen")
            os.remove(temp_path)
            return jsonify({"error": "No se pudo cargar la imagen"}), 400

        logger.info("Convertir la imagen de BGR a RGB")
        # Convertir la imagen de BGR a RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        logger.info("Procesando la imagen para detección de puntos clave")
        # Procesar la imagen para detección de puntos clave
        results = face_mesh.process(image_rgb)

        ojeras = arrugas = None

        # Procesar los rostros detectados
        if results.multi_face_landmarks:
            logger.info("Detectando rostros")
            for face_landmarks in results.multi_face_landmarks:
                h, w, _ = image.shape

                eye_indices_profile = [53, 160, 445, 355]
                face_indices_profile = [10, 234, 454, 150]

                eye_points_profile = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in eye_indices_profile]
                face_points_profile = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in face_indices_profile]

                eye_roi_profile = image[min(p[1] for p in eye_points_profile):max(p[1] for p in eye_points_profile),
                                        min(p[0] for p in eye_points_profile):max(p[0] for p in eye_points_profile)]

                face_roi_profile = image[min(p[1] for p in face_points_profile):max(p[1] for p in face_points_profile),
                                         min(p[0] for p in face_points_profile):max(p[0] for p in face_points_profile)]

                ojeras = detect_dark_areas(eye_roi_profile)
                arrugas = detect_wrinkles(face_roi_profile)

                # Análisis del porcentaje
                if 0 <= int(ojeras) <= 5 or 0 <= int(arrugas) <= 10:
                    estado = "Normal"
                elif 6 <= int(ojeras) <= 10 or 11 <= int(arrugas) <= 15:
                    estado = "Falta de sueño o estrés"
                elif 11 <= int(ojeras) <= 20 or 16 <= int(arrugas) <= 25:
                    estado = "Consumo moderado"
                else:
                    estado = "Consumo alto"

                promedio = (ojeras + arrugas) / 2

                # Limpiar archivo temporal
                os.remove(temp_path)

                logger.info(f"Ojeras: {ojeras}, Arrugas: {arrugas}, Promedio: {promedio}, Estado: {estado}")

                return jsonify({
                    "ojeras": round(ojeras, 2),
                    "arrugas": round(arrugas, 2),
                    "promedio": round(promedio, 2),
                    "estado": estado
                })

        os.remove(temp_path)
        logger.error("No se detectaron rostros en la imagen")
        return jsonify({"error": "No se detectaron rostros en la imagen"}), 400

    except Exception as e:
        logger.exception("Ocurrió un error durante el procesamiento")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

