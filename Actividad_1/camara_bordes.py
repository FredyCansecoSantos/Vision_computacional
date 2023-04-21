
# Este codigo fue hecho en colaboracion con el profesor Cesar Torres Huitzil
import cv2
import numpy as np
import time

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
    scale_rgb_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
    scale_rgb_frame = scale_rgb_frame[:, :, :: -1]
    return frame, scale_rgb_frame

def show_frame(name, frame):
    # Muestra la imagen resultante
    cv2.imshow(name,frame)

lastPublication = 0.0
PUBLISH_TIME = 10


video_capture = init_camera()

while (True):
    bgr_frame, scale_rgb_frame = acquire_image(video_capture)

    if np.abs(time.time()-lastPublication) > PUBLISH_TIME:
        try:
            print("No remote action needed ...")
        except (keyboardInterrup):
            break
        except Exception as e:
            print(e)
        lastPublication = time.time()

    img_gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(img_gray,50,200)
    bgr_frame[edges == 255] = [125,249,39]


    show_frame ('RGB image', bgr_frame)
    show_frame ('Gray level image', img_gray)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
