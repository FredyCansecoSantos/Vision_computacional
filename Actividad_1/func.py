from skimage.color import gray2rgb
import skimage.io as io
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage import filters
from skimage import feature
import skimage
import numpy as np
import random

#----------------------------------------------
def rgb_and_gray_scale(imagen):
    # Transformar la imagen en escala de grises
    img_grayscale_4 = io.imread('road54.png', as_gray=True)
    img_grayscale_5 = gray2rgb(img_grayscale_4)

    figura, ejes = plt.subplots(1, 2, figsize=(8, 4))
    ejs = ejes.ravel()
    ejs[0].imshow(imagen)
    ejs[0].set_title("Color")
    ejs[1].imshow(img_grayscale_5)
    ejs[1].set_title("Escala de gris")

    figura.tight_layout()
    plt.show()
#-------------------------------------------------
def binarizacion(imagen):
    # Convertir la imagen a escala de grises
    img_binarizada = skimage.color.rgb2gray(imagen)

    # Binarizar la imagen utilizando un umbral fijo (en este caso, 0.2)
    Umbral = 0.2
    img_binarizada = img_binarizada > Umbral

    # Se muestra la imagen binarizada
    skimage.io.imshow(img_binarizada)
    skimage.io.show()
#-------------------------------------------------

def filtro_gaussiano_and_blur(imagen):
  #se convierten a escala de grises
  escala_grises = io.imread('road55.png', as_gray=True)
  img_grayscale_5 = gray2rgb(escala_grises)
  ruido = skimage.util.random_noise(img_grayscale_5, mode='gaussian', seed=None, clip=True)

  imagen_con_blur = filters.gaussian(ruido, sigma=2, multichannel=True)


  figura, ejes = plt.subplots(1, 2, figsize=(8, 4))
  ejs = ejes.ravel()

  ejs[0].imshow(ruido)
  ejs[0].set_title("Ruido Gaussiano")
  ejs[1].imshow(imagen_con_blur)
  ejs[1].set_title('Imagen filtrada')

  plt.tight_layout()
  plt.show()


#-------------------------------------------------
def deteccion_bordes(imagen):
    escala_grises = io.imread('road55.png', as_gray=True)
    imagen_gris = skimage.color.rgb2gray(imagen)
    # Detectar bordes en la imagen en escala de grises
    #bordes = filters.canny(imagen_gris)
    bordes = feature.canny(imagen_gris)

    figura, ejes = plt.subplots(1, 2, figsize=(8, 4))
    ejs = ejes.ravel()

    ejs[0].imshow(imagen)
    ejs[0].set_title("Imagen normal")
    ejs[1].imshow(bordes)
    ejs[1].set_title('Deteccion de bordes')

    plt.tight_layout()
    plt.show()
#-------------------------------------------------

imagen = io.imread('road55.png')

#rgb_and_gray_scale(imagen)
#binarizacion(imagen)
#filtro_gaussiano_and_blur(imagen)
deteccion_bordes(imagen)
