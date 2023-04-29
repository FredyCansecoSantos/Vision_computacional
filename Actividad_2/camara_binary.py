# Este codigo fue hecho en colaboracion con José Ángel Ramírez Ramírez
import cv2
import numpy as np
import time

from skimage.filters import rank, threshold_sauvola
from skimage.util import img_as_ubyte
from skimage.morphology import disk
from skimage.feature import peak_local_max
from scipy import ndimage

from scipy import ndimage as ndi

IMG_ROW_RES = 480
IMG_COL_RES = 640

def init_camera():
    video_capture = cv2.VideoCapture(0)
    ret = video_capture.set(3, IMG_COL_RES)
    ret = video_capture.set(4, IMG_ROW_RES)
    return video_capture

def acquire_image(video_capture):
    # Utiliza un solo frame de video
    ret, frame = video_capture.read()
    scale_rgb_frame = cv2.resize(frame, (0,0), fx=1, fy=1)
    gray_frame = cv2.cvtColor(scale_rgb_frame, cv2.COLOR_BGR2GRAY)
    return frame, scale_rgb_frame, gray_frame

def show_frame(name, frame):
    # Muestra la imagen resultante
    cv2.imshow(name,frame)

lastPublication = 0.0
PUBLISH_TIME = 10

video_capture = init_camera()

while (True):
    bgr_frame, scale_rgb_frame, gray_frame = acquire_image(video_capture)

    if np.abs(time.time()-lastPublication) > PUBLISH_TIME:
        try:
            print("No remote action needed ...")
        except (KeyboardInterrupt):
            break
        except Exception as e:
            print(e)
        lastPublication = time.time()

    denoised = rank.median(img_as_ubyte(gray_frame), disk(5))
    markers = rank.gradient(denoised,disk(5)) < 10
    markers = ndi.label(markers)[0]
    gradient = rank.gradient(denoised,disk(2))
    threshold_value, binary_otsu = cv2.threshold(gray_frame, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    labels = ndi.label(binary_otsu)[0]

    show_frame ('RGB image', bgr_frame)
    show_frame ('Markers', markers.astype(np.int8))
    show_frame ('RGB watershed', cv2.applyColorMap(labels.astype(np.uint8),cv2.COLORMAP_JET))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
