import streamlit as st
import cv2
import joblib
import imutils
import numpy as np
from PIL import Image
from ultralytics import YOLO
from metricas_disco_escavacao import CDR, RDR, NRR, BVR, BVR2 , CDRvh, excentricidade
from xgboost import XGBClassifier

def crop_image_with_margin(image, box, margin=10):
    width, height = image.size
    x1, y1, x2, y2 = box
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(width, x2 + margin)
    y2 = min(height, y2 + margin)
    return image.crop((x1, y1, x2, y2))



def contours_(x_min, y_min, x_max, y_max):
    # Calcular o centro da caixa delimitadora
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2

    # Calcular os semi-eixos da elipse
    semi_axis_x = (x_max - x_min) / 2
    semi_axis_y = (y_max - y_min) / 2

    return center_x, center_y, semi_axis_x, semi_axis_y

def draw_contours(image, result_disc, result_cup):
    new_img = image.copy()
    altura, largura, _ = image.shape
    mask_disc = np.zeros((altura, largura), dtype=np.uint8)
    mask_cup = np.zeros((altura, largura), dtype=np.uint8)

    # Considerar apenas a região de maior confiança
    if len(result_disc.boxes) > 0:
        best_disc = max(result_disc.boxes, key=lambda x: x.conf[0])
        x_min, y_min, x_max, y_max = map(int, best_disc.xyxy[0].tolist())
        center_x, center_y, semi_axis_x, semi_axis_y = contours_(x_min, y_min, x_max, y_max)
        cv2.ellipse(image, (int(center_x), int(center_y)), (int(semi_axis_x), int(semi_axis_y)), 0, 0, 360, (0, 255, 0), 2)
        cv2.ellipse(mask_disc, (int(center_x), int(center_y)), (int(semi_axis_x), int(semi_axis_y)), 0, 0, 360, (255, 255, 255), -1)
        cnts_disc = cv2.findContours(mask_disc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts_disc = imutils.grab_contours(cnts_disc)
        cnt_disc = max(cnts_disc, key=cv2.contourArea)

    if len(result_cup.boxes) > 0:
        best_cup = max(result_cup.boxes, key=lambda x: x.conf[0])
        x_min2, y_min2, x_max2, y_max2 = map(int, best_cup.xyxy[0].tolist())
        center_x2, center_y2, semi_axis_x2, semi_axis_y2 = contours_(x_min2, y_min2, x_max2, y_max2)
        cv2.ellipse(image, (int(center_x2), int(center_y2)), (int(semi_axis_x2), int(semi_axis_y2)), 0, 0, 360, (0, 255, 0), 2)
        cv2.ellipse(mask_cup, (int(center_x2), int(center_y2)), (int(semi_axis_x2), int(semi_axis_y2)), 0, 0, 360, (255, 255, 255), -1)
        cnts_cup = cv2.findContours(mask_cup, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts_cup = imutils.grab_contours(cnts_cup)
        cnt_cup = min(cnts_cup, key=cv2.contourArea)

    cdr = CDR(cnt_disc, cnt_cup)
    cdrv, cdrh = CDRvh(cnt_disc, cnt_cup)
    rdr = RDR(cnt_disc, cnt_cup)
    nrr = NRR(cnt_disc, cnt_cup, new_img)
    return cdr, cdrv, cdrh, rdr, nrr



def img_retina(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # pixels pretos
    black_pixels = np.sum(np.all(image_rgb == [0, 0, 0], axis=2))
    # número total de pixels
    total_pixels = image_rgb.shape[0] * image_rgb.shape[1]
    # proporção de pixels pretos
    black_pixel_ratio = black_pixels / total_pixels

    if black_pixel_ratio > 0.1:
        results_cup = yolo_model_cup(image, conf=0.01)
        # Verificar se há detecções
        if results_cup and len(results_cup[0].boxes.xyxy) > 0:
            # Encontrar a detecção com a maior confiança
            max_confidence = 0
            best_box = None
            for box, conf in zip(results_cup[0].boxes.xyxy, results_cup[0].boxes.conf):
                if conf > max_confidence:
                    max_confidence = conf
                    best_box = box

            # Se uma detecção com confiança máxima foi encontrada
            if best_box is not None:
                box = best_box.cpu().numpy().astype(int).tolist()
                
                # Recortar a imagem com a margem
                image_final = crop_image_with_margin(image, box, 10)
    else:
        image_final = image

    return image_final
# Função principal para processar a imagem
def process_image(image_bytes, yolo_model_disc, yolo_model_cup, confidence_threshold=0.01):
    # Converter bytes para imagem
    image_in = Image.open(image_bytes)
    # image = image.resize((640,640))
    image = img_retina(image_in)
    image_np = np.array(image)

    # Processar imagem com os modelos YOLO para discos e copos
    results_disc = yolo_model_disc(image, conf=confidence_threshold)
    results_cup = yolo_model_cup(image, conf=confidence_threshold)

    # Desenhar retângulos em volta das regiões detectadas
    cdr, cdrv, cdrh, rdr, nrr = draw_contours(image_np, results_disc[0], results_cup[0])

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