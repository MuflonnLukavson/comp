import cv2
import time
import numpy as np
import json
from itertools import combinations

cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("http://10.100.33.246:4747/video") ## Change the IP for connecting the phone camera
cap = cv2.VideoCapture("http://192.168.1.51:4747/video") 
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
    "lines": [],
    "homography, warp": [],
    "total": [],
}    

dst_H, dst_W =  2 * 297,2* 210  # A4 size
destination_pts = np.array([[0, 0], [dst_W-1, 0], [dst_W-1, dst_H-1], [0, dst_H-1]], dtype=np.float32)

def get_lines(points):
    lines = []
    for point in points:
        # point [x1, y1, x2, y2]
        x1, y1, x2, y2 = point
        line = np.linalg.cross(np.array([x1, y1, 1]), np.array([x2, y2, 1]))
        lines.append(line)
    return lines

def intersect_lines(l1, l2):
    # l1 and l2 are in homogeneous coordinates
    x = np.cross(l1, l2)
    if x[2] == 0:
        return None  # lines are parallel
    return (x[0]/x[2], x[1]/x[2])  # convert back to Cartesian coordinates


def dlt_estimate(points_src, points_dst):
    A = []
    for (x, y), (u, v) in zip(points_src, points_dst):
        A.append([0, 0, 0, -x, -y, -1, v*x, v*y, v])
        A.append([-x, -y, -1, 0, 0, 0, u*x, u*y, u])
    A = np.array(A)
    U, S, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)
    return H / H[2, 2]  # Normalize 

border = False

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

    img_blur = cv2.blur(gray, (3,3))
    detected_edges = cv2.Canny(img_blur, low_threshold, low_threshold*ratio, kernel_size)
    # get all points 
    ys, xs = np.nonzero(detected_edges)
    points = np.column_stack((xs, ys))

    t2 = time.time() - t1 - t0

    line_points = cv2.HoughLinesP(detected_edges, 1, np.pi/180, 200, minLineLength=150, maxLineGap=50)
    # print(line_points)
    # 4 Strongest lines
    t3 = time.time() - t2 - t1 - t0

    if line_points is not None:

        strongest_lines = sorted(line_points, key=lambda x: np.linalg.norm(x[0:2] - x[2:4]), reverse=True)[:4]


        lines = get_lines(strongest_lines)
        intersections = []
        for l1, l2 in combinations(lines, 2):
            pt = intersect_lines(l1, l2)
            if pt is not None and 0 <= pt[0] < W and 0 <= pt[1] < H:
                intersections.append(pt)

        if not intersections == None:
            intersections = sorted(intersections[:4], key=lambda x: np.linalg.norm(x))

        ## order the points in clockwise order
        if len(intersections) == 4:
            pts = np.array(intersections, dtype=np.float32)
            s = pts.sum(axis=1) # sum of x and y coordinates
            d = np.diff(pts, axis=1).ravel() # difference between x and y coordinates
            source_pts = np.array([pts[np.argmin(s)],    
                                pts[np.argmin(d)],    
                                pts[np.argmax(s)],    
                                pts[np.argmax(d)]],  
                                dtype=np.float32)

            if cv2.contourArea(source_pts) > 1000:
                ## compute homography
                source_pts = np.array(source_pts, dtype=np.float32)       
                destination_pts = np.array(destination_pts, dtype=np.float32)

                t4 = time.time() - t3 - t2 - t1 - t0

                h, status = cv2.findHomography(source_pts, destination_pts,)
                # h = dlt_estimate(source_pts, destination_pts)


                if h is not None:
                    im_out = cv2.warpPerspective(frame, h, ((dst_W, dst_H)))
                    warp = True
                t5 = time.time() - t4 - t3 - t2 - t1 - t0
        else:
            warp = False
            t4 = np.nan
            t5 = np.nan



        for line in strongest_lines:
            x1, y1, x2, y2 = line
            cv2.line(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

        for pt in intersections:
            x, y = int(pt[0]), int(pt[1])
            cv2.circle(display, (x, y), 5, (0, 0, 255), -1)
    else:
        t4 = 0
        t5 = 0
        warp = False



    times["capture"].append(t1)
    times["edges"].append(t2)
    times["hough"].append(t3)
    times["lines"].append(t4)
    times["homography, warp"].append(t5)
    times["total"].append(time.time() - t0)

    cv2.putText(display, f'fps: {fps:.1f}', (10, 30), font, 1, (0, 255, 255), 2, cv2.LINE_4)

    if border:
        display = detected_edges

    cv2.imshow('frame',display)
    if warp:
        cv2.imshow('warped', im_out)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('b'):
        border = not border

print("Average times:")
for key, value in times.items():
    arr = np.asarray(value, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size:
        print(f"{key}: {arr.mean()*1000:.1f} ms (std {arr.std()*1000:.1f}, n={arr.size})")
    else:
        print(f"{key}: no valid samples")