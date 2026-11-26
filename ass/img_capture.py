from pathlib import Path

import cv2


cap = cv2.VideoCapture("http://10.100.33.246:4747/video")
output_dir = Path(__file__).parent / "img"
output_dir.mkdir(exist_ok=True)

if not cap.isOpened():
    cap.release()
    raise RuntimeError("Could not open the camera or stream")

existing_numbers = [
    int(path.stem[3:])
    for path in output_dir.glob("img*.jpg")
    if path.stem[3:].isdigit()
]
image_number = max(existing_numbers, default=-1) + 1

print("Press Enter to save the current frame. Press q to quit.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("Could not read a frame from the camera or stream")

        cv2.imshow("Camera", frame)
        key = cv2.waitKey(1) & 0xFF

        if key in (10, 13):
            image_path = output_dir / f"img{image_number:02d}.jpg"
            if not cv2.imwrite(str(image_path), frame):
                raise OSError(f"Could not save image to {image_path}")
            print(f"Saved {image_path}")
            image_number += 1
        elif key == ord("q"):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
