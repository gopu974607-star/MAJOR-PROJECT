from ultralytics import YOLO
import cv2

model = YOLO(r"runs\detect\runs\final_pen_leaf_v2\weights\best.pt")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

print("Show PEN or LEAF")
print("Press S to capture")
print("Press Q to quit")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Camera error")
        break

    cv2.imshow("PEN + LEAF TEST", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        cv2.imwrite("object_test.jpg", frame)
        print("Saved object_test.jpg")
        break

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()