
from ultralytics import YOLO
import cv2
import os
import time
import math
import json


# ============================================================
#                    MODEL PATH
# ============================================================

YOLO_MODEL_PATH = (
    r"C:\Users\diyad\OneDrive\Desktop\YOLO_Project"
    r"\runs\detect\train-16\weights\best.pt"
)


# ============================================================
#                    CAMERA SETTINGS
# ============================================================

CAMERA_INDEX = 0

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

IMAGE_SIZE = 640


# ============================================================
#                    DETECTION SETTINGS
# ============================================================

YOLO_CONFIDENCE = 0.55
IOU_THRESHOLD = 0.45


# ============================================================
#                    CAMERA CALIBRATION
# ============================================================

FX = 700.0
FY = 700.0

CX = FRAME_WIDTH / 2
CY = FRAME_HEIGHT / 2

FOCAL_LENGTH = 700.0


# ============================================================
#                 CAMERA -> ROBOT OFFSET
# ============================================================

TX = 0.0
TY = 0.0
TZ = 0.0


# ============================================================
#                 REAL OBJECT WIDTHS
#                 IN CENTIMETERS
# ============================================================

OBJECT_WIDTHS = {

    "person": 40.0,
    "bicycle": 60.0,
    "car": 180.0,
    "motorcycle": 80.0,
    "bus": 250.0,
    "truck": 250.0,

    "bottle": 6.0,
    "cup": 8.0,
    "book": 15.0,
    "cell phone": 7.0,
    "laptop": 30.0,
}

DEFAULT_OBJECT_WIDTH = 10.0


# ============================================================
#                         COLORS
# ============================================================

COLORS = {

    "person":      (255, 0, 0),
    "bicycle":     (0, 255, 255),
    "car":         (0, 165, 255),
    "motorcycle":  (255, 0, 255),
    "bus":         (255, 255, 0),
    "truck":       (128, 0, 255),

    "bottle":      (0, 255, 0),
    "cup":         (255, 128, 0),
    "book":        (0, 128, 255),
    "cell phone":  (128, 255, 0),
    "laptop":      (255, 0, 128),
}


FALLBACK_COLORS = [
    (255, 0, 0),
    (0, 255, 255),
    (0, 165, 255),
    (255, 0, 255),
    (255, 255, 0),
    (128, 0, 255),
    (255, 128, 0),
    (128, 255, 0),
    (255, 0, 128)
]

fallback_color_index = 0


def get_color(name):

    global fallback_color_index

    if name in COLORS:
        return COLORS[name]

    color = FALLBACK_COLORS[
        fallback_color_index % len(FALLBACK_COLORS)
    ]

    fallback_color_index += 1

    return color


# ============================================================
#                    CHECK MODEL
# ============================================================

if not os.path.isfile(YOLO_MODEL_PATH):

    print("\n" + "=" * 60)
    print("MODEL NOT FOUND")
    print("=" * 60)
    print(YOLO_MODEL_PATH)
    print("=" * 60)

    raise SystemExit


# ============================================================
#                    LOAD YOLO MODEL
# ============================================================

print("\nLoading YOLO model...")

yolo_model = YOLO(YOLO_MODEL_PATH)


print("\n" + "=" * 60)
print("MODEL LOADED SUCCESSFULLY")
print("=" * 60)

print("MODEL PATH:")
print(YOLO_MODEL_PATH)

print("\nYOLO CLASSES:")
print(yolo_model.names)

print("=" * 60)


# ============================================================
#              DISTANCE ESTIMATION
# ============================================================

def estimate_distance(object_name, pixel_width):

    if pixel_width <= 0:
        return 0.0

    real_width = OBJECT_WIDTHS.get(
        object_name,
        DEFAULT_OBJECT_WIDTH
    )

    distance_cm = (
        real_width * FOCAL_LENGTH
    ) / pixel_width

    return distance_cm


# ============================================================
#              CAMERA XYZ
# ============================================================

def pixel_to_camera_xyz(
    center_x,
    center_y,
    distance_cm
):

    Z = distance_cm / 100.0

    X = (
        (center_x - CX) * Z
    ) / FX

    Y = (
        (center_y - CY) * Z
    ) / FY

    return X, Y, Z


# ============================================================
#              CAMERA -> ROBOT XYZ
# ============================================================

def camera_to_robot(
    camera_x,
    camera_y,
    camera_z
):

    robot_x = camera_x + TX
    robot_y = camera_y + TY
    robot_z = camera_z + TZ

    return robot_x, robot_y, robot_z


# ============================================================
#                    POSITION
# ============================================================

def get_position(
    center_x,
    center_y,
    width,
    height
):

    if center_x < width / 3:
        horizontal = "Left"

    elif center_x < (2 * width / 3):
        horizontal = "Center"

    else:
        horizontal = "Right"


    if center_y < height / 3:
        vertical = "Top"

    elif center_y < (2 * height / 3):
        vertical = "Middle"

    else:
        vertical = "Bottom"


    return horizontal, vertical


# ============================================================
#                    DRAW DETECTION
# ============================================================

latest_detections = []


def draw_detection(
    frame,
    object_name,
    confidence,
    x1,
    y1,
    x2,
    y2
):

    global latest_detections

    frame_height, frame_width = frame.shape[:2]

    # --------------------------------------------------------
    # Keep coordinates inside frame
    # --------------------------------------------------------

    x1 = max(
        0,
        min(x1, frame_width - 1)
    )

    y1 = max(
        0,
        min(y1, frame_height - 1)
    )

    x2 = max(
        0,
        min(x2, frame_width - 1)
    )

    y2 = max(
        0,
        min(y2, frame_height - 1)
    )


    if x2 <= x1 or y2 <= y1:
        return


    # --------------------------------------------------------
    # CENTER
    # --------------------------------------------------------

    center_x = int(
        (x1 + x2) / 2
    )

    center_y = int(
        (y1 + y2) / 2
    )


    # --------------------------------------------------------
    # BOX SIZE
    # --------------------------------------------------------

    pixel_width = x2 - x1
    pixel_height = y2 - y1

    area = pixel_width * pixel_height


    # --------------------------------------------------------
    # DISTANCE
    # --------------------------------------------------------

    distance_cm = estimate_distance(
        object_name,
        pixel_width
    )


    # --------------------------------------------------------
    # CAMERA XYZ
    # --------------------------------------------------------

    camera_x, camera_y, camera_z = (
        pixel_to_camera_xyz(
            center_x,
            center_y,
            distance_cm
        )
    )


    # --------------------------------------------------------
    # ROBOT XYZ
    # --------------------------------------------------------

    robot_x, robot_y, robot_z = (
        camera_to_robot(
            camera_x,
            camera_y,
            camera_z
        )
    )


    # --------------------------------------------------------
    # POSITION
    # --------------------------------------------------------

    horizontal, vertical = get_position(
        center_x,
        center_y,
        frame_width,
        frame_height
    )


    # --------------------------------------------------------
    # COLOR
    # --------------------------------------------------------

    color = get_color(object_name)


    # ========================================================
    # DRAW BOX
    # ========================================================

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        2
    )


    # ========================================================
    # DRAW GRASP CENTER
    # ========================================================

    cv2.circle(
        frame,
        (center_x, center_y),
        5,
        color,
        -1
    )


    cv2.line(
        frame,
        (center_x - 9, center_y),
        (center_x + 9, center_y),
        color,
        1
    )

    cv2.line(
        frame,
        (center_x, center_y - 9),
        (center_x, center_y + 9),
        color,
        1
    )


    # ========================================================
    # LABEL
    # ========================================================

    confidence_percent = confidence * 100.0

    label = (
        f"{object_name.upper()} "
        f"{confidence_percent:.1f}%"
    )

    font = cv2.FONT_HERSHEY_SIMPLEX

    font_scale = 0.48
    thickness = 2


    (
        text_width,
        text_height
    ), baseline = cv2.getTextSize(
        label,
        font,
        font_scale,
        thickness
    )


    label_top = max(
        2,
        y1 - text_height - 8
    )


    cv2.rectangle(
        frame,
        (x1, label_top),
        (
            x1 + text_width + 8,
            label_top + text_height + 8
        ),
        color,
        -1
    )


    cv2.putText(
        frame,
        label,
        (
            x1 + 4,
            label_top + text_height + 2
        ),
        font,
        font_scale,
        (0, 0, 0),
        thickness
    )


    # ========================================================
    # INFORMATION
    # ========================================================

    info_y = y2 + 17

    if info_y + 55 >= frame_height:

        info_y = max(
            y1 + 18,
            18
        )


    cv2.putText(
        frame,
        f"Center: ({center_x}, {center_y})",
        (x1, info_y),
        font,
        0.40,
        color,
        1
    )


    cv2.putText(
        frame,
        f"Box: {pixel_width}x{pixel_height}px",
        (x1, info_y + 15),
        font,
        0.40,
        color,
        1
    )


    cv2.putText(
        frame,
        f"Distance: {distance_cm:.1f} cm",
        (x1, info_y + 30),
        font,
        0.40,
        color,
        1
    )


    cv2.putText(
        frame,
        f"Position: {horizontal}, {vertical}",
        (x1, info_y + 45),
        font,
        0.40,
        color,
        1
    )


    # ========================================================
    # STORE DATA
    # ========================================================

    object_data = {

        "object": object_name,

        "confidence_percent":
            round(
                confidence_percent,
                2
            ),

        "bounding_box": {

            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2
        },

        "grasp_center": {

            "x": center_x,
            "y": center_y
        },

        "pixel_width":
            pixel_width,

        "pixel_height":
            pixel_height,

        "area_pixels":
            area,

        "distance_cm":
            round(
                distance_cm,
                2
            ),

        "position": {

            "horizontal": horizontal,
            "vertical": vertical
        },

        "camera_xyz_m": {

            "x": round(camera_x, 4),
            "y": round(camera_y, 4),
            "z": round(camera_z, 4)
        },

        "robot_xyz_m": {

            "x": round(robot_x, 4),
            "y": round(robot_y, 4),
            "z": round(robot_z, 4)
        }
    }


    latest_detections.append(
        object_data
    )


# ============================================================
#                    OPEN WEBCAM
# ============================================================

print("\nOpening webcam...")

cap = cv2.VideoCapture(
    CAMERA_INDEX,
    cv2.CAP_DSHOW
)


if not cap.isOpened():

    print("\nERROR: Webcam could not be opened.")

    raise SystemExit


cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    FRAME_WIDTH
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    FRAME_HEIGHT
)

cap.set(
    cv2.CAP_PROP_FPS,
    30
)


# ============================================================
#                    FPS
# ============================================================

previous_time = time.time()

fps = 0.0


# ============================================================
#                    START
# ============================================================

print("\n" + "=" * 60)
print("WEBCAM DETECTION STARTED")
print("=" * 60)

print("General YOLO : ACTIVE")
print("\nPress Q to quit")
print("=" * 60)


# ============================================================
#                    MAIN LOOP
# ============================================================

while True:

    # ========================================================
    # READ FRAME
    # ========================================================

    success, frame = cap.read()

    if not success:

        print("Could not read camera frame.")

        break


    # ========================================================
    # ORIGINAL FRAME
    # ========================================================

    original_frame = frame.copy()

    latest_detections = []


    # ========================================================
    # GENERAL YOLO
    # ========================================================

    yolo_results = yolo_model.predict(

        source=original_frame,

        conf=YOLO_CONFIDENCE,

        iou=IOU_THRESHOLD,

        imgsz=IMAGE_SIZE,

        max_det=30,

        verbose=False
    )


    # ========================================================
    # PROCESS DETECTIONS
    # ========================================================

    for result in yolo_results:

        if result.boxes is None:
            continue


        for box in result.boxes:

            confidence = float(
                box.conf[0]
            )

            class_id = int(
                box.cls[0]
            )


            # ------------------------------------------------
            # Check class
            # ------------------------------------------------

            if class_id not in yolo_model.names:
                continue


            object_name = (
                yolo_model.names[class_id]
            )


            # ------------------------------------------------
            # Bounding box
            # ------------------------------------------------

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )


            # ------------------------------------------------
            # Draw everything detected by original YOLO
            # ------------------------------------------------

            draw_detection(

                frame,

                object_name,

                confidence,

                x1,
                y1,
                x2,
                y2
            )


    # ========================================================
    # FPS
    # ========================================================

    current_time = time.time()

    elapsed = (
        current_time -
        previous_time
    )


    if elapsed > 0:

        instant_fps = 1.0 / elapsed

        fps = (
            0.9 * fps +
            0.1 * instant_fps
        )


    previous_time = current_time


    # ========================================================
    # FPS DISPLAY
    # ========================================================

    cv2.putText(

        frame,

        f"FPS: {fps:.1f}",

        (10, 25),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.60,

        (255, 255, 255),

        2
    )


    # ========================================================
    # OBJECT COUNT
    # ========================================================

    cv2.putText(

        frame,

        f"Objects: {len(latest_detections)}",

        (10, 50),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.55,

        (255, 255, 255),

        2
    )


    # ========================================================
    # SAVE JSON
    # ========================================================

    coordinate_packet = {

        "timestamp":
            time.time(),

        "objects":
            latest_detections
    }


    try:

        with open(
            "object_coordinates.json",
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                coordinate_packet,
                file,
                indent=2
            )

    except Exception:

        pass


    # ========================================================
    # PRINT DETECTIONS
    # ========================================================

    if latest_detections:

        print("\n" + "-" * 60)


        for obj in latest_detections:

            print(
                f"OBJECT       : "
                f"{obj['object']}"
            )

            print(
                f"CONFIDENCE   : "
                f"{obj['confidence_percent']:.2f}%"
            )


            bbox = obj["bounding_box"]

            print(
                f"BOUNDING BOX : "
                f"({bbox['x1']}, "
                f"{bbox['y1']}, "
                f"{bbox['x2']}, "
                f"{bbox['y2']})"
            )


            center = obj["grasp_center"]

            print(
                f"GRASP CENTER : "
                f"({center['x']}, "
                f"{center['y']})"
            )


            print(
                f"DISTANCE     : "
                f"{obj['distance_cm']:.2f} cm"
            )


            position = obj["position"]

            print(
                f"POSITION     : "
                f"{position['horizontal']}, "
                f"{position['vertical']}"
            )


            camera = obj["camera_xyz_m"]

            print(
                f"CAMERA XYZ   : "
                f"({camera['x']:.3f}, "
                f"{camera['y']:.3f}, "
                f"{camera['z']:.3f}) m"
            )


            robot = obj["robot_xyz_m"]

            print(
                f"ROBOT XYZ    : "
                f"({robot['x']:.3f}, "
                f"{robot['y']:.3f}, "
                f"{robot['z']:.3f}) m"
            )


        print("-" * 60)


    # ========================================================
    # SHOW
    # ========================================================

    cv2.imshow(
        "Robotic Hand Object Detection",
        frame
    )


    # ========================================================
    # QUIT
    # ========================================================

    key = cv2.waitKey(1) & 0xFF


    if key == ord("q"):
        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()


print("\n" + "=" * 60)
print("PROGRAM FINISHED")
print("=" * 60)

