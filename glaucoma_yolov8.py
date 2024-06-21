import streamlit as st
import cv2
import joblib
import imutils
import numpy as np
from PIL import Image
from ultralytics import YOLO
from metricas_disco_escavacao import CDR, RDR, NRR, BVR, BVR2 , CDRvh, excentricidade
from xgboost import XGBClassifier


def contours_(x_min, y_min, x_max, y_max):
    # Calcular o centro da caixa delimitadora
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2

    # Calcular os semi-eixos da elipse
    semi_axis_x = (x_max - x_min) / 2
    semi_axis_y = (y_max - y_min) / 2

    return center_x, center_y, semi_axis_x, semi_axis_y

def draw_contours(image, results_disc, results_cup):
    new_img = image.copy()
    altura, largura, _ = image.shape
    mask_disc = np.zeros((altura, largura), dtype=np.uint8)

    for result in results_disc:
        for bbox in result.boxes.xyxy:
            x_min, y_min, x_max, y_max = map(int, bbox.tolist())
            # Desenhar a elipse na imagem
            center_x, center_y, semi_axis_x, semi_axis_y = contours_(x_min, y_min, x_max, y_max)
            cv2.ellipse(image, (int(center_x), int(center_y)), (int(semi_axis_x), int(semi_axis_y)), 0, 0, 360, (0,  255,0) , 2)
            cv2.ellipse(mask_disc, (int(center_x), int(center_y)), (int(semi_axis_x), int(semi_axis_y)), 0, 0, 360, (255,255,255), -1)
            cnts_disc = cv2.findContours(mask_disc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts_disc = imutils.grab_contours(cnts_disc)
            cnt_disc = max(cnts_disc, key=cv2.contourArea)
    mask_cup = np.zeros((altura, largura), dtype=np.uint8)
    for result2 in results_cup:
        for bbox2 in result2.boxes.xyxy:
            x_min2, y_min2, x_max2, y_max2 = map(int, bbox2.tolist())

            center_x2, center_y2, semi_axis_x2, semi_axis_y2 = contours_(x_min2, y_min2, x_max2, y_max2)
            cv2.ellipse(image, (int(center_x2), int(center_y2)), (int(semi_axis_x2), int(semi_axis_y2)), 0, 0, 360, (0, 255,0) , 2)
            cv2.ellipse(mask_cup, (int(center_x2), int(center_y2)), (int(semi_axis_x2), int(semi_axis_y2)), 0, 0, 360, (255,255,255), -1)

            cnts_cup = cv2.findContours(mask_cup, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts_cup = imutils.grab_contours(cnts_cup)
            cnt_cup = min(cnts_cup, key=cv2.contourArea)

    cdr = CDR(cnt_disc, cnt_cup)
    cdrv, cdrh = CDRvh(cnt_disc, cnt_cup)
    rdr = RDR(cnt_disc, cnt_cup)
    nrr = NRR(cnt_disc, cnt_cup, new_img)
    return cdr, cdrv, cdrh, rdr, nrr



# Função principal para processar a imagem
def process_image(image_bytes, yolo_model_disc, yolo_model_cup, confidence_threshold=0.1):
    # Converter bytes para imagem
    image = Image.open(image_bytes)
    # image = image.resize((640,640))
    image_np = np.array(image)

    # Processar imagem com os modelos YOLO para discos e copos
    results_disc = yolo_model_disc(image, conf=confidence_threshold)
    results_cup = yolo_model_cup(image, conf=confidence_threshold)

    # Desenhar retângulos em volta das regiões detectadas
    cdr, cdrv, cdrh, rdr,nrr =  draw_contours(image_np, results_disc, results_cup)

    pred = loaded_model.predict([[cdr, cdrv, cdrh, rdr,nrr]])
    if pred == 0:
        res = 'Normal'
    elif pred==1:
        res = 'Glaucoma'

    return image_np, cdr, cdrv, cdrh, rdr,nrr, res

# Carregar modelos YOLO
yolo_model_disc = YOLO('models/disc.pt')
yolo_model_cup = YOLO('models/cup.pt')
loaded_model = joblib.load('models/model_xgb.pkl')

# Configurações do aplicativo Streamlit
st.title("Detect Glaucoma in Optic Nerve")
image_bytes = st.file_uploader("Select an image", type=["jpg", "jpeg", "png"])

# Verificar se uma imagem foi carregada
if image_bytes is not None:
    # Processar imagem com modelos YOLO para discos e copos
    output_image, cdr, cdrv, cdrh, rdr,nrr, res = process_image(image_bytes, yolo_model_disc, yolo_model_cup)

    col1, col2, col3 = st.columns(3)

    # Exibir imagem com retângulos em volta das regiões detectadas
    col1.image(image_bytes, caption='Original image', width=200)
    col2.image(output_image, caption='Optic Disc and Cup Segmentation', width=200)
    # col3.image(image_bytes, caption='Imagem Original', width=output_image.shape[1] // 3)
    # Exibir métricas na terceira coluna
    col3.header("Métricas")
    if cdr is not None:
        col3.write(f"CDR: {cdr}")
    if cdrv is not None:
        col3.write(f"CDRv: {cdrv}")
    if cdrh is not None:
        col3.write(f"CDRh: {cdrh}")
    if rdr is not None:
        col3.write(f"RDR: {rdr}")
    if nrr is not None:
        col3.write(f"NRR: {nrr}")
    if res is not None:
        col3.write(f"Result: {res}")