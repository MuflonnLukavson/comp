import cv2
import time
import numpy as np
from torch import classes
import json
from itertools import combinations


cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("http://10.100.33.246:4747/video") ## Change the IP for connecting the phone camera
font = cv2.FONT_HERSHEY_SIMPLEX

if not cap.isOpened():
    raise RuntimeError("Could not open the camera or stream")


t1_fps = 0
t2_fps = 0
ret, frame = cap.read()


H, W, C = frame.shape
fps = 30
count = 0

low_threshold = 30
ratio = 5
kernel_size = 3

times = {
    "capture": [],
    "edges": [],
    "hough": [],
    "total": [],
}


def ang_diff(t1, t2):                       # angle between two lines, 0..pi/2
    d = abs(t1 - t2) % np.pi
    return min(d, np.pi - d)

def similar(l1, l2, d_rho=30, d_theta=np.radians(10)):
    (r1, t1), (r2, t2) = l1, l2
    if ang_diff(t1, t2) < d_theta and abs(r1 - r2) < d_rho:
        return True
    return False

def intersect(l1, l2):
    (r1, t1), (r2, t2) = l1, l2
    A = np.array([[np.cos(t1), np.sin(t1)], [np.cos(t2), np.sin(t2)]])
    if abs(np.linalg.det(A)) < 1e-6:
        return None
    return np.linalg.solve(A, [r1, r2])      # (x, y)

def find_lines(lines, W, H, max_cands=8, par_deg=20, perp_deg=30, margin=50):
    # 1. non-maximum suppression: keep the strongest of each cluster of duplicates
    cands = []
    for l in lines[:, 0]:
        if not any(similar(l, c) for c in cands):
            cands.append(l)
        if len(cands) == max_cands:
            break

    
    combos = sorted(combinations(range(len(cands)), 4), key=sum)
    for combo in combos:
        q = [cands[i] for i in combo]
        for pairs in (((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))):
            (a, b), (c, d) = pairs
            if (ang_diff(q[a][1], q[b][1]) > np.radians(par_deg) or
                ang_diff(q[c][1], q[d][1]) > np.radians(par_deg)):
                continue                     
            if any(ang_diff(q[i][1], q[j][1]) < np.radians(perp_deg)
                   for i in (a, b) for j in (c, d)):
                continue                     
            pts = [intersect(q[i], q[j]) for i in (a, b) for j in (c, d)]
            if any(p is None or not (-margin < p[0] < W + margin and
                                      -margin < p[1] < H + margin) for p in pts):
                continue
            return q, np.array(pts)
    return None, None



while(True):
    t0 = time.time()
    ret, frame = cap.read()
    t1 = time.time() - t0
    count += 1

    # frame = cv2.resize(frame, (H // 2, W // 2))
    
    if count == 5:
        t1_fps = t2_fps
        t2_fps = time.time() 
        fps = 5 / (t2_fps - t1_fps)
        count = 0


    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    display = frame.copy()

    img_blur = cv2.blur(gray, (4,4))
    detected_edges = cv2.Canny(img_blur, low_threshold, low_threshold*ratio, kernel_size)
    # get all points 
    ys, xs = np.nonzero(detected_edges)
    points = np.column_stack((xs, ys))

    t2 = time.time() - t1 - t0

    lines = cv2.HoughLines(detected_edges, 1, np.pi/180, 150)
    # print(line_points)
    # 4 Strongest lines


    if lines is not None:
        quad, corners = find_lines(lines, W, H)
        t3 = time.time() - t2 - t1 - t0



    # for line in strongest_lines:
    #     x1, y1, x2, y2 = line
    #     cv2.line(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

    times["capture"].append(t1)
    times["edges"].append(t2)
    times["hough"].append(t3)
    times["total"].append(time.time() - t0)

    # display = detected_edges
    cv2.putText(display, f'fps: {fps:.1f}', (10, 30), font, 1, (0, 255, 255), 2, cv2.LINE_4)
    
    if quad is not None:
        L = 3000
        for rho, theta in quad:
            x0, y0 = rho * np.cos(theta), rho * np.sin(theta)
            dx, dy = -np.sin(theta), np.cos(theta)
            cv2.line(display, (int(x0 - L*dx), int(y0 - L*dy)),
                            (int(x0 + L*dx), int(y0 + L*dy)), (0, 255, 0), 2)
        for x, y in corners:
            cv2.circle(display, (int(x), int(y)), 8, (0, 0, 255), -1)

    cv2.imshow('frame',display)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("Average times:")
for key, value in times.items():
    print(f"{key}: {np.mean(value)*1000:.1f} ms")