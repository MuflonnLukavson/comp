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
fps = 30
count = 0


low_threshold = 18
ratio = 3
kernel_size = 3

RANSAC_T = 4

while(True):
    start_time = time.time()
    ret, frame = cap.read()
    capture_time = time.time() - start_time
    count += 1

    # frame = cv2.resize(frame, (H // 2, W // 2))

    if count == 5:
        t1 = t2
        t2 = time.time() 
        fps = 5 / (t2 - t1)
        count = 0


    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    display = frame.copy()

    ## Makes it less noisy
    img_blur = cv2.blur(gray, (3,3))
    detected_edges = cv2.Canny(img_blur, low_threshold, low_threshold*ratio, kernel_size)
    # get all points 
    ys, xs = np.nonzero(detected_edges)
    points = np.column_stack((xs, ys))
    # take every 4th
    # points = points[::4]

    edges_time = time.time() - capture_time - start_time


    ## === RANSAC ===
    N = 500
    best_fit = None
    best_fit_inliers = -1
    for _ in range(N):
        ## Choose two random point for ransac
        p1 = points[np.random.randint(0, len(points))]
        p2 = points[np.random.randint(0, len(points))]
        # print(p1, p2)
            
        a = float(p2[1] - p1[1])
        b = float(p1[0] - p2[0])
        norm = np.hypot(a, b)
        if norm == 0:                       # identical coordinates, skip
            continue
        a, b = a / norm, b / norm
        c = -(a * p1[0] + b * p1[1])

        dist = np.abs(points @ np.array([a, b]) + c)
        inliers = np.count_nonzero(dist < RANSAC_T)

        if best_fit_inliers < inliers:
            best_fit_inliers = inliers
            best_fit = (a, b ,c)

    ransac_time = time.time() - edges_time - start_time - capture_time
 
    # refit using LSQ
    a, b, c = best_fit
    inliers = points[np.abs(points @ np.array([a, b]) + c) < RANSAC_T].astype(np.float32)

    vx, vy, x0, y0 = cv2.fitLine(inliers, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    L = 2000
    q1 = (int(x0 - L * vx), int(y0 - L * vy))
    q2 = (int(x0 + L * vx), int(y0 + L * vy))
    cv2.line(display, q1, q2, (0, 0, 255), 2)

    refit_time = time.time() - ransac_time - start_time  - edges_time - capture_time

    ## === Print the time taken for each step ===
    # print(f"Capture: {capture_time*1000:.1f}, Edges: {edges_time*1000:.1f}, RANSAC: {ransac_time*1000:.1f}, Refit: {refit_time*1000 :.1f}")
    # print(f'Number of endpoints: {len(points)}, Outliers ratio: {1 - best_fit_inliers/len(points):.2f}')

    cv2.putText(display, f'fps: {fps:.1f}', (10, 30), font, 1, (0, 255, 255), 2, cv2.LINE_4)

    a, b, c = best_fit
    x0, y0 = -a * c, -b * c          # closest point to origin
    dx, dy = -b, a                   # direction
    q1 = (int(x0 - L * dx), int(y0 - L * dy))
    q2 = (int(x0 + L * dx), int(y0 + L * dy))
    cv2.line(display, q1, q2, (0, 255, 0), 2)



    # display = detected_edges
    cv2.imshow('frame',display)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



cap.release()
cv2.destroyAllWindows()