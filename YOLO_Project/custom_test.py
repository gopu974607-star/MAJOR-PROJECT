from ultralytics import YOLO
import cv2

model = YOLO(
    r"C:\Users\diyad\OneDrive\Desktop\YOLO_Project\runs\detect\runs\final_pen_leaf_v2\weights\best.pt"
)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("ERROR: Camera not opened")
    exit()

print("CUSTOM MODEL TEST")
print("Show a PEN or LEAF")
print("Press Q to quit")

while True:
    ret, frame = cap.read()

    if not ret:
        print("ERROR: Cannot read camera")
        break

    results = model.predict(
        source=frame,
        conf=0.10,
        verbose=False,
        device=0
    )

    for result in results:
        for box in result.boxes:

            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            name = model.names[class_id]

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            print(
                f"{name}  confidence={confidence:.3f}"
            )

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"{name} {confidence:.2f}",
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

    cv2.imshow("Pen + Leaf Custom Model Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()