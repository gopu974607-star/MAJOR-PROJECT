import cv2

# Open webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("ERROR: Camera could not be opened")
    exit()

print("===================================")
print("PEN IMAGE CAPTURE")
print("Show the pen clearly to the camera")
print("Press S = Save")
print("Press Q = Quit")
print("===================================")

while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read camera")
        break

    cv2.imshow("Capture Pen", frame)

    key = cv2.waitKey(1) & 0xFF

    # Save image
    if key == ord("s"):

        save_path = r"C:\Users\diyad\OneDrive\Desktop\pen_test.jpg"

        success = cv2.imwrite(save_path, frame)

        if success:
            print("SUCCESS!")
            print("Saved image to:")
            print(save_path)
        else:
            print("ERROR: Image could not be saved")

        break

    # Quit
    if key == ord("q"):
        print("Quit")
        break

cap.release()
cv2.destroyAllWindows()