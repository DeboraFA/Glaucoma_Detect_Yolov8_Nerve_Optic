
from skimage.morphology import disk, black_tophat
import cv2
import numpy as np


def CDR(disc, cup):
    #### CDR ####
    disc_area = cv2.contourArea(disc)
    cup_area = cv2.contourArea(cup)
    
    CDR = cup_area/disc_area
    
    return CDR

def CDRvh(cnt_disc, cnt_cup):
    # CDR horizontal e vertical 
    # Calcular o retângulo delimitador do disco óptico (vertical e horizontal)
    x_disc, y_disc, diameter_horizontal_disc, diameter_vertical_disc = cv2.boundingRect(cnt_disc)

    # Calcular o retângulo delimitador da escavação (vertical e horizontal)
    x_cup, y_cup, diameter_horizontal_cup, diameter_vertical_cup = cv2.boundingRect(cnt_cup)

    # Calcular o CDR vertical
    CDR_vertical = diameter_vertical_cup / diameter_vertical_disc

    # Calcular o CDR horizontal
    CDR_horizontal = diameter_horizontal_cup / diameter_horizontal_disc

    return CDR_vertical, CDR_horizontal

def excentricidade(cnt_cup):
    # Excentricidade
    # Calcular elipse ajustada à escavação
    ellipse = cv2.fitEllipse(cnt_cup)

    # Calcular eixo maior e eixo menor da elipse
    major_axis = max(ellipse[1])
    minor_axis = min(ellipse[1])

    # Calcular excentricidade
    eccentricity = np.sqrt(1 - (minor_axis / major_axis) ** 2)
    return eccentricity




def RDR(cnt_disc, cnt_cup):

    disc_area = cv2.contourArea(cnt_disc)
    cup_area = cv2.contourArea(cnt_cup)

    area_rim =  disc_area - cup_area
    RDR = area_rim/disc_area
    RDR
    
    return RDR

def NRR(disc,cup,img):

    x,y,z = np.shape(img)
    maskSup, maskNasal, maskInf, maskTemp = Create_Masks(x,y)

    mask = np.zeros((x,y), np.uint8)
    disc = cv2.drawContours(mask,[disc],0,[255,255,255],-1)

    mask = np.zeros((x,y), np.uint8)
    cup = cv2.drawContours(mask,[cup],0,[255,255,255],-1)

    nrr = cv2.bitwise_xor(disc,cup)

    nrrSup = cv2.bitwise_and(nrr,maskSup)

    contours_nrrSup, hierarchy_nrrSup = cv2.findContours(nrrSup, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)        
    if np.shape(hierarchy_nrrSup) == (0,):
        nrrSup_area = 0
    else:
        nrrSup_area = cv2.contourArea(max(contours_nrrSup, key=cv2.contourArea))

    nrrNasal = cv2.bitwise_and(nrr,maskNasal)

    contours_nrrNasal, hierarchy_nrrNasal = cv2.findContours(nrrNasal, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if np.shape(hierarchy_nrrNasal) == (0,):
        nrrNasal_area = 1
    else:
        nrrNasal_area = cv2.contourArea(max(contours_nrrNasal, key=cv2.contourArea))


    nrrInf = cv2.bitwise_and(nrr,maskInf)

    contours_nrrInf, hierarchy_nrrInf = cv2.findContours(nrrInf, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if np.shape(hierarchy_nrrInf) == (0,):
        nrrInf_area = 0
    else:
        nrrInf_area = cv2.contourArea(max(contours_nrrInf, key=cv2.contourArea))

    nrrTemp = cv2.bitwise_and(nrr,maskTemp)
    contours_nrrTemp, hierarchy_nrrTemp = cv2.findContours(nrrTemp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if np.shape(hierarchy_nrrTemp) == (0,):
        nrrTemp_area = 1
    else:
        nrrTemp_area = cv2.contourArea(max(contours_nrrTemp, key=cv2.contourArea))

    NRR = (nrrInf_area + nrrSup_area)/(nrrNasal_area + nrrTemp_area)

    return NRR
    
    
def BVR(disc_cnt,img):
    #### Mudar para a de Gerar_imagens
    x,y,z = np.shape(img)
    
    mask = np.zeros((x,y), np.uint8)
    disc_img = cv2.drawContours(mask,[disc_cnt],0,[255,255,255],-1)
    
    r,g,b = cv2.split(img)
    
    g = cv2.bitwise_and(g,disc_img)
    
    maskSup, maskNasal, maskInf, maskTemp = Create_Masks(x,y)
    
    gBT = black_tophat(g, disk(20))
    ret2,th2 = cv2.threshold(gBT,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    imT = cv2.medianBlur(th2, 3)
    
    bvrSup = cv2.bitwise_and(imT,maskSup)
    bvrSupArea = np.sum(bvrSup == 255)

    bvrNasal = cv2.bitwise_and(imT,maskNasal)
    bvrNasalArea = np.sum(bvrNasal == 255)

    bvrInf = cv2.bitwise_and(imT,maskInf)
    bvrInfArea = np.sum(bvrInf == 255)

    bvrTemp = cv2.bitwise_and(imT,maskTemp)
    bvrTempArea = np.sum(bvrTemp == 255)

    bvr = (bvrInfArea + bvrSupArea)/(bvrNasalArea + bvrTempArea)
      
    return bvr

def BVR2(cnt_disc, img):
    
    x,y,z = np.shape(img)
    mask = np.zeros((x,y), np.uint8)
    disc_img = cv2.drawContours(mask,[cnt_disc],0,[255,255,255],-1)
    
    r,g,b = cv2.split(img)
    
    g = cv2.bitwise_and(g,disc_img)
    
    gBT = black_tophat(g, disk(20))
    ret2,th2 = cv2.threshold(gBT,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    imT = cv2.medianBlur(th2, 3)
    
    # Calcular a área total da retina
    total_area = cv2.contourArea(cnt_disc)

    # Calcular a área dos vasos sanguíneos (exemplo hipotético usando contagem de pixels)
    vessel_area = np.count_nonzero(imT)

    # Calcular o BVR
    bvr2 = (vessel_area / total_area) 
    return bvr2


def Create_Masks(x,y):
    x = x
    y = y
    mask = np.zeros((x,y), np.uint8)
    triangle_temp = np.array( [(x, 0), (x, y), (int(x/2), int(y/2)) ])
    maskTemp = cv2.drawContours(mask, [triangle_temp], 0, (255,255,255), -1)

    rows,cols = maskTemp.shape

    M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
    maskSup = cv2.warpAffine(maskTemp,M,(cols,rows))

    M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
    maskNasal = cv2.warpAffine(maskSup,M,(cols,rows))

    M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
    maskInf = cv2.warpAffine(maskNasal,M,(cols,rows))

    M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
    maskTemp = cv2.warpAffine(maskInf,M,(cols,rows))
    
    return maskSup, maskNasal, maskInf, maskTemp