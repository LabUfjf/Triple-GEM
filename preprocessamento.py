# -*- coding: utf-8 -*-

import cv2

#import operator
import numpy as np
#import sys
import h5py

RESIZE_LARGURA_IMG = 600;
RESIZE_ALTURA_IMG = 600;

DEBUG = 1
RESIZE = 0

numRun = '00060'
numImg = '0'

#numRun = '494'
#numImg = '1'

def carregaImagem(numRun, numImg):
    url = 'h5files/histograms_Run' + numRun + '.hdf5';
    #url = 'Run' + numRun + '/Data_Image/I' + numImg + 'Run' + numRun + 'S.png';
    print('URL da imagem: ' + url)
    #img = cv2.imread(url)
    f = h5py.File('C:/Users/Igor/Desktop/Triple-GEM/histograms_Run00060.hdf5','r')
    img1 = f['pic_run' + numRun +'_ev' + numImg][:]
    img = np.array(img1,dtype = 'uint8')
    f.close()
    
    if RESIZE:
        img = cv2.resize(img, (RESIZE_ALTURA_IMG, RESIZE_LARGURA_IMG))             # Redimensiona para melhor visualização
    
    if DEBUG:
        cv2.imshow("imgOriginal", img)
    return img
    
def filtraImagem(img):
    imgFiltrada = cv2.fastNlMeansDenoising(img,None,20,7,21) # 1
    
    mimg = np.mean(img);
    
    
#    imgFiltrada = cv2.bilateralFilter(img,9,75,75) # 2
#    kernel = np.ones((6,6),np.float32)/36 # 3
#    imgFiltrada = cv2.filter2D(img,-1,kernel) # 3
#    imgFiltrada = cv2.GaussianBlur(imgFiltrada,(5,5),0) # 4
#    imgFiltrada = cv2.medianBlur(img,5)
    return imgFiltrada
    
def imagemThreshold(img):
    img = cv2.GaussianBlur(img,(9,9),0)
#    ret,imgThresh = cv2.threshold(img,0,255,cv2.THRESH_OTSU)
    ret,imgThresh = cv2.threshold(img,115,255,cv2.THRESH_BINARY)            # TESTE: threshold global, gerou um resultado melhor que o adaptativo
#    imgThresh = cv2.adaptiveThreshold(img,                                   # imagem de entrada
#                                      255,                                  # pixels que passam pelo threshold tornam-se totalmente brancos
#                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,       # usar gaussiana ao invés de média, gera um melhor resultado
#                                      cv2.THRESH_BINARY_INV,                    # não inverte o fundo
#                                      21,                                   # tamanho da vizinhança de pixels para calculo do threshold
#                                      2)                                    # constante subtraída da média ou da média ponderada
    
    return imgThresh
def skeleton(img):
    edges = cv2.Canny(img,10,50,apertureSize = 3)
    minLineLength = 5
    maxLineGap = 100
    lines = cv2.HoughLinesP(img,1,np.pi/180,100,minLineLength,maxLineGap)
    if(lines.all() != None):
        print('OK, foram encontradas linhas na imagem')
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
    return img
def maximizeContrast(imgGrayscale):

    height, width = imgGrayscale.shape

    imgTopHat = np.zeros((height, width, 1), np.uint8)
    imgBlackHat = np.zeros((height, width, 1), np.uint8)

    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    imgTopHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_TOPHAT, structuringElement)
    imgBlackHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_BLACKHAT, structuringElement)

    imgGrayscalePlusTopHat = cv2.add(imgGrayscale, imgTopHat)
    imgGrayscalePlusTopHatMinusBlackHat = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)

    return imgGrayscalePlusTopHatMinusBlackHat
def extractValue(imgOriginal):
    height, width, numChannels = imgOriginal.shape

    imgHSV = np.zeros((height, width, 3), np.uint8)

    imgHSV = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2HSV)

    imgHue, imgSaturation, imgValue = cv2.split(imgHSV)

    return imgValue
# end function

def extraiContornos(img, imgThresh):
    im2,contours,hierarchy = cv2.findContours(imgThresh, 1, 2)
    for i in range(len(contours)):        
        cnt = contours[i]
        rect = cv2.minAreaRect(cnt)
        #if rect[1][0] > 100 or rect[1][1] > 100:
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(img,[box],0,(0,0,255),2)
    return img
#        M = cv2.moments(contours[i])
#        print M
#        if(M['m00'] != 0):
#            cx = int(M['m10']/M['m00'])
#            cy = int(M['m01']/M['m00'])
#            img[cx][cy] = (0,255,0);

def copiaThreshNaImagemOriginal(img, imgThresh):
    imgCopia = img
    for x in range(0, imgThresh.shape[0], 1):
        for y in range(0, imgThresh.shape[1], 1):
            ponto = imgThresh[x][y]
            if ponto == 255:
                imgCopia[x][y] = 255
    if DEBUG:
        cv2.imwrite( "debug/imgNova_run" + numRun + "_" + numImg + ".jpg", imgCopia)
    imgFiltrada = filtraImagem(imgCopia)
    #imgFiltrada = cv2.cvtColor(imgFiltrada, cv2.COLOR_BGR2GRAY)                           # pega a imagem em escala de cinzas
    if DEBUG:
        cv2.imwrite( "debug/imgFiltrada2_run" + numRun + "_" + numImg + ".jpg", imgFiltrada)
    print('Calculando o threshold na imagem novamente... ')
    imgThresh = imagemThreshold(imgFiltrada)
    if DEBUG:
        cv2.imshow("imgThreshNovo", imgThresh)
        cv2.imwrite( "debug/imgThresh2_run" + numRun + "_" + numImg + ".jpg", imgThresh)
    return imgThresh

def avaliaTracos(img):
    for x in range(0, img.shape[0], 1):
        for y in range(0, img.shape[1], 1):
            m = img[x-1][y+1] * mascara[0][0]
            m += img[x-1][y] * mascara[1][0]
            m += img[x-1][y-1] * mascara[2][0]
def realcaTracos(img):
#    b = 64. # brightness
#    c = 0.  # contrast
#    return cv2.addWeighted(img, 1. + c/127., img, 0, b-c)
    imghsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    imghsv[:,:,2] = [[max(pixel - 25, 0) if pixel < 190 else min(pixel + 25, 255) for pixel in row] for row in imghsv[:,:,2]]
    return cv2.cvtColor(imghsv, cv2.COLOR_HSV2BGR)

    cv2.imshow(img_corr)

def clusteringKmeans(img):
    Z = img.reshape((-1,3)) 
    # convert to np.float32
    Z = np.float32(Z)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 6
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
 
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    cv2.imshow('res2',res2)
    return res2

if __name__ == '__main__':
    
    print('Carregando a imagem... ')
    img = carregaImagem(numRun, numImg)
    imgCopia = img.copy()
#    imgCopia = realcaTracos(imgCopia)
#    imgGrayscale = extractValue(img)
#    imgFiltrada = cv2.bilateralFilter(img,9,75,75)
#    cv2.imshow("imgTeste", imgTeste)
    print('Filtrando a imagem... ')
    imgFiltrada = filtraImagem(imgCopia)
#    imgFiltrada = clusteringKmeans(imgFiltrada)
    #imgFiltrada = cv2.cvtColor(imgFiltrada, cv2.COLOR_BGR2GRAY)                           # pega a imagem em escala de cinzas    

#    imgFiltrada = maximizeContrast(imgFiltrada)
    if DEBUG:
        cv2.imshow("imgFiltrada", imgFiltrada)
        cv2.imwrite( "debug/imgFiltrada_run" + numRun + "_" + numImg + ".jpg", imgFiltrada)
    print('Calculando o threshold na imagem... ')
    imgThresh = imagemThreshold(imgFiltrada)
    if DEBUG:
        cv2.imshow("imgThresh", imgThresh)
        cv2.imwrite( "debug/imgThresh1_run" + numRun + "_" + numImg + ".jpg", imgThresh)
    imgThresh = copiaThreshNaImagemOriginal(imgCopia, imgThresh)
    img = extraiContornos(img, imgThresh)    
    if DEBUG:
        cv2.imshow("imgFinal", img)
    cv2.imwrite( "imagens/result_run" + numRun + "_" + numImg + ".jpg", img)
#    dst    = cv2.thinning(imgThresh, thinningType = THINNING_ZHANGSUEN)
#    cv2.imshow("dst", dst)
    cv2.waitKey(0)