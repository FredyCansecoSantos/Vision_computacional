#Este código fue implementado en colaboración con Ángel Ramírez Ramírez
import cv2
import numpy as np
import matplotlib.pyplot as plt

def threshold_func(imagen_gris):

  ret, th_bin_img = cv2.threshold(imagen_gris, 127, 255, cv2.THRESH_BINARY_INV)
  
  th_adp_img = cv2.adaptiveThreshold(imagen_gris, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 7, 5) 
 
  th_sauvola =  cv2.adaptiveThreshold(imagen_gris, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25, -10)

  return th_bin_img, th_adp_img, th_sauvola
  
for i in range(52,57):
  ruta_imagen = './road'+str(i)+".png"
  img = cv2.imread(ruta_imagen)

  gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  th_bin_img, th_adp_img, th_sauvola = threshold_func(gray_img)


  fig, axs = plt.subplots(1,3)
  
  axs[0].axis('off')
  axs[0].title.set_text('Binaria')
  axs[0].imshow(th_bin_img, cmap ='gray')

  axs[1].axis('off')
  axs[1].title.set_text('Adaptativa')
  axs[1].imshow(th_adp_img, cmap ='gray')

  axs[2].axis('off')
  axs[2].title.set_text('Sauvola')
  axs[2].imshow(th_sauvola, cmap ='gray')
  plt.show()

def watersheed_func_2(imagen, imagen_gris):

  ret, th_bin_img = cv2.threshold(imagen_gris, 127, 255, cv2.THRESH_BINARY_INV)
  SE = np.ones((15,15), np.uint8)
  ero_img = cv2.erode(th_bin_img, SE, iterations = 1)


  ret, markers = cv2.connectedComponents(ero_img)
  markers = cv2.watershed(imagen, markers)
  imagen[markers == -1] = [255,0,0]
  
  return imagen, markers

for i in range(52,57):
  ruta = "./road"+str(i)+".png"
  img = cv2.imread(ruta)
  img =  cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  imagen, markers = watersheed_func_2(img, gray_img)

  fig, axs = plt.subplots(1,2)
  
  axs[0].axis('off')
  axs[0].title.set_text('segmentación Watersheed')
  axs[0].imshow(imagen)

  axs[1].axis('off')
  axs[1].title.set_text('Regiones Segmentadas')
  axs[1].imshow(markers, cmap='bone')

  def ruido_gauss(img, level=1):
    img_noisy = np.zeros(img.shape, np.uint8)
    
    if level == 1:
        std = 10
    elif level == 2:
        std = 50
    elif level == 3:
        std = 100
    else:

        return None

    for i in range(3):
        noise = np.zeros(img.shape[:2], np.uint8)
        cv2.randn(noise, 0, std)
        img_noisy[:,:,i] = cv2.add(img[:,:,i], noise)

    return img_noisy


for i in range(52,57):

  fig, axs = plt.subplots(1,10)
  ruta = "./road"+str(i)+".png"
  img = cv2.imread(ruta)
  img =  cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

  #Agrega ruido gaussiano a la imagen en 2 niveles 2 y 3
  img_con_ruido2 = ruido_gauss(img, level=2)
  img_con_ruido3 = ruido_gauss(img, level=3)
  #convierte la imagen en gris y le aplica las 3 segmentacionse(global, local y Sauvola)
  img_gris1 = cv2.cvtColor(img_con_ruido2, cv2.COLOR_RGB2GRAY)
  img_gris2 = cv2.cvtColor(img_con_ruido3, cv2.COLOR_RGB2GRAY)
  th_bin_img_noise, th_adp_img_noise, th_sauvola_noise = threshold_func(img_gris1)
  th_bin_img_noise2, th_adp_img_noise2, th_sauvola_noise2 = threshold_func(img_gris2)
  #segmentacion por watersheed
  imagen, markers = watersheed_func_2(img_con_ruido2, img_gris1)
  imagen1, markers1 = watersheed_func_2(img_con_ruido3, img_gris2)

  axs[0].title.set_text('RN1')
  axs[0].imshow(img_con_ruido2)
  axs[0].axis('off')

  axs[1].title.set_text('RN2')
  axs[1].imshow(img_con_ruido3)
  axs[1].axis('off')

  axs[2].title.set_text('Bin')
  axs[2].imshow(th_bin_img_noise, cmap='gray')
  axs[2].axis('off')

  axs[3].title.set_text('Adap')
  axs[3].imshow(th_adp_img_noise, cmap='gray')
  axs[3].axis('off')

  axs[4].title.set_text('sauv')
  axs[4].imshow(th_sauvola_noise, cmap='gray')
  axs[4].axis('off')

  axs[5].title.set_text('Bin')
  axs[5].imshow(th_bin_img_noise2, cmap='gray')
  axs[5].axis('off')

  axs[6].title.set_text('Adap')
  axs[6].imshow(th_adp_img_noise2, cmap='gray')
  axs[6].axis('off')

  axs[7].title.set_text('sauv')
  axs[7].imshow(th_sauvola_noise2, cmap='gray')
  axs[7].axis('off')

  axs[8].axis('off')
  axs[8].title.set_text('WS')
  axs[8].imshow(imagen)

  axs[9].axis('off')
  axs[9].title.set_text('SR')
  axs[9].imshow(markers, cmap='bone')