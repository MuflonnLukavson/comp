import json, cv2, numpy as np
d = json.load(open("bh-phone-annotation.json"))
v = next(v for v in d["_via_img_metadata"].values() if v["filename"] == "126.jpg")
img = cv2.imread("images/bh-phone/126.jpg")
# img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)   # or ROTATE_90_COUNTERCLOCKWISE
for r in v["regions"]:
    s = r["shape_attributes"]
    pts = np.array(list(zip(s["all_points_x"], s["all_points_y"])), np.int32)
    cv2.polylines(img, [pts], True, (0, 255, 0), 4)
cv2.imwrite("check_126.jpg", cv2.resize(img, None, fx=0.25, fy=0.25))
