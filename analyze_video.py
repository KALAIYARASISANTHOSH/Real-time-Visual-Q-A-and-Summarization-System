import cv2
import json
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


# ==========================================
# CONFIG
# ==========================================

MODEL_PATH = "model/yolo11s.pt"
VIDEO_PATH = "test4.avi"
JSON_PATH = "video_data.json"

CONFIDENCE = 0.5


# ==========================================
# LOAD MODELS
# ==========================================

print("Loading YOLO...")
model = YOLO(MODEL_PATH)

print("Loading Deep SORT...")
tracker = DeepSort(
    max_age=30,
    n_init=3,
    max_cosine_distance=0.2,
    nn_budget=None
)


# ==========================================
# OPEN VIDEO
# ==========================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError("Could not open video")


fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

duration = total_frames / fps if fps > 0 else 0

print("FPS:", fps)
print("Frames:", total_frames)
print("Duration:", duration)


# ==========================================
# OBJECT DATABASE
# ==========================================

objects = {}

frame_number = 0


# ==========================================
# DISPLAY WINDOW
# ==========================================

cv2.namedWindow(
    "YOLO11 + Deep SORT",
    cv2.WINDOW_NORMAL
)


# ==========================================
# PROCESS VIDEO
# ==========================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    timestamp = frame_number / fps


    # ======================================
    # YOLO DETECTION
    # ======================================

    results = model(
        frame,
        conf=CONFIDENCE,
        verbose=False
    )

    detections = []

    for box in results[0].boxes:

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

        confidence = float(box.conf[0])

        class_id = int(box.cls[0])

        class_name = model.names[class_id]

        detections.append(
            (
                [
                    float(x1),
                    float(y1),
                    float(x2 - x1),
                    float(y2 - y1)
                ],
                confidence,
                class_name
            )
        )


    # ======================================
    # DEEP SORT
    # ======================================

    tracks = tracker.update_tracks(
        detections,
        frame=frame
    )


    # ======================================
    # DRAW TRACKS + SAVE JSON DATA
    # ======================================

    for track in tracks:

        if not track.is_confirmed():
            continue

        track_id = str(track.track_id)

        class_name = track.get_det_class()

        if class_name is None:
            continue


        # ----------------------------------
        # Save object information
        # ----------------------------------

        if track_id not in objects:

            objects[track_id] = {

                "track_id": int(track_id),

                "class": class_name,

                "first_seen": round(
                    timestamp,
                    2
                ),

                "last_seen": round(
                    timestamp,
                    2
                ),

                "timestamps": []

            }


        objects[track_id][
            "last_seen"
        ] = round(timestamp, 2)


        objects[track_id][
            "timestamps"
        ].append(
            round(timestamp, 2)
        )


        # ----------------------------------
        # Bounding box
        # ----------------------------------

        l, t, r, b = track.to_ltrb()

        l = int(l)
        t = int(t)
        r = int(r)
        b = int(b)


        # ----------------------------------
        # Class colors
        # ----------------------------------

        class_colors = {

            "person": (0, 0, 255),

            "car": (255, 0, 0),

            "motorcycle": (0, 255, 0),

            "bus": (0, 255, 255),

            "truck": (255, 0, 255),

            "bicycle": (255, 255, 0)

        }

        color = class_colors.get(
            class_name,
            (255, 255, 255)
        )


        # ----------------------------------
        # Draw bounding box
        # ----------------------------------

        cv2.rectangle(

            frame,

            (l, t),

            (r, b),

            color,

            2
        )


        # ----------------------------------
        # Draw label
        # ----------------------------------

        label = (
            f"{class_name} "
            f"ID:{track_id}"
        )

        cv2.putText(

            frame,

            label,

            (l, max(t - 10, 20)),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.6,

            color,

            2
        )


    # ======================================
    # SHOW VIDEO
    # ======================================

    cv2.imshow(
        "YOLO11 + Deep SORT",
        frame
    )


    # ======================================
    # QUIT
    # ======================================

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):

        print("Stopped by user")

        break


# ==========================================
# RELEASE VIDEO
# ==========================================

cap.release()

cv2.destroyAllWindows()


# ==========================================
# CALCULATE DURATIONS
# ==========================================

for obj in objects.values():

    obj["duration"] = round(

        obj["last_seen"]
        -
        obj["first_seen"],

        2
    )


# ==========================================
# OBJECT SUMMARY
# ==========================================

class_counts = {}

for obj in objects.values():

    class_name = obj["class"]

    class_counts[class_name] = (

        class_counts.get(
            class_name,
            0
        )
        + 1

    )


# ==========================================
# CREATE JSON
# ==========================================

video_data = {

    "video": VIDEO_PATH,

    "fps": fps,

    "total_frames": total_frames,

    "duration": round(
        duration,
        2
    ),

    "object_summary": class_counts,

    "objects": list(
        objects.values()
    )

}


# ==========================================
# SAVE JSON
# ==========================================

with open(
    JSON_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        video_data,
        f,
        indent=2
    )


print()
print("==============================")
print("Video analysis completed")
print("JSON:", JSON_PATH)
print("Objects:", len(objects))
print("==============================")
