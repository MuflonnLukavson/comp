import cv2
import time
import numpy as np

cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("http://10.100.33.246:4747/video") ## Change the IP for connecting the phone camera
font = cv2.FONT_HERSHEY_SIMPLEX

if not cap.isOpened():
    raise RuntimeError("Could not open the camera or stream")


t1 = 0
t2 = 0
ret, frame = cap.read()


H, W, C = frame.shape
print(H, W, C)
count = 0
fps = 30
timer_start = time.time()

while(True):
    ret, frame = cap.read()
    count += 1
    if count == 5:
        t1 = t2
        t2 = time.time() 
        fps = 5 / (t2 - t1)
        count = 0


    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    display = frame.copy()



    ## ==== Getting the brightest spot ====
    # For Loop
    # value = 0
    # max_loc = None
    # for i in range(H):
    #     for j in range(W):
    #         new_val = gray[i, j]
    #         if new_val > value:
    #             value = new_val
    #             max_loc = (j, i)

    # Numpy arrays
    # idx = np.argmax(gray)
    # max_loc = divmod(idx, W)[::-1] 

    # OpenCV
    max_loc = cv2.minMaxLoc(gray)[3]

    ## === Reddest spot ===
    # r - (b + g) / 2 
    # b, g, r = cv2.split(frame)  
    # redness = r - (b + g) / 2
    # idx = np.argmax(redness)
    # reddest_pos = divmod(idx, W)[::-1] 


    # Lab representation
    lab_img = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(lab_img)
    idx = np.argmax(a)
    reddest_pos2 = divmod(idx, W)[::-1] 


    ## Timer
    # time_ms = -(timer_start - time.time()) * 1000

    ## Adding shapes to displayed image
    # cv2.putText(display, f'{time_ms:.0f}ms', (960, 60), font, 2, (0, 165, 255), 2, cv2.LINE_4)

    cv2.putText(display, f'fps: {fps:.1f}', (10, 30), font, 1, (0, 255, 255), 2, cv2.LINE_4)

    cv2.circle(display, max_loc, 10, (255, 0, 0), 2)

    # cv2.circle(display, reddest_pos, 10, (0, 0, 255), 2)
    cv2.circle(display, reddest_pos2, 10, (0, 255, 255), 2)




    cv2.imshow('frame',display)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



cap.release()
cv2.destroyAllWindows()


