import cv2
from ultralytics import YOLO

# ==========================================
# BOTTLE CAP INSPECTOR
# ==========================================

MODEL_PATH = "best.pt"
VIDEO_PATH = "bottle_video.mp4"
OUTPUT_PATH = "outputs/bottle_inspection.mp4"

# Load YOLO model
model = YOLO(MODEL_PATH)

# Open video
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Error: Could not open video")
    exit()

# Video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

if fps == 0:
    fps = 25

# Output video
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(
    OUTPUT_PATH,
    fourcc,
    fps,
    (width, height)
)

# Class names
# Change these according to your trained model
CAP_CLASSES = ["capped", "cap"]
UNCAP_CLASSES = ["uncapped", "no_cap", "uncap"]

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # YOLO detection
    results = model(frame, conf=0.40, verbose=False)

    total = 0
    capped = 0
    uncapped = 0

    for result in results:

        boxes = result.boxes

        for box in boxes:

            # Coordinates
            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )

            # Confidence
            confidence = float(box.conf[0])

            # Class ID
            class_id = int(box.cls[0])

            class_name = model.names[class_id].lower()

            total += 1

            # ----------------------------------
            # CAP CLASSIFICATION
            # ----------------------------------

            if class_name in CAP_CLASSES:

                capped += 1

                color = (0, 255, 0)
                status = "CAPPED"

            else:

                uncapped += 1

                color = (0, 0, 255)
                status = "UNCAPPED"

            # ----------------------------------
            # Draw bounding box
            # ----------------------------------

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                3
            )

            # Label
            label = f"{status} {confidence * 100:.0f}%"

            # Label background
            (tw, th), _ = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                2
            )

            cv2.rectangle(
                frame,
                (x1, y1 - th - 12),
                (x1 + tw + 10, y1),
                color,
                -1
            )

            # Label text
            cv2.putText(
                frame,
                label,
                (x1 + 5, y1 - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2
            )

    # ==========================================
    # INSPECTION DASHBOARD
    # ==========================================

    panel_x = 20
    panel_y = 20
    panel_w = 310
    panel_h = 155

    # Transparent panel
    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (panel_x, panel_y),
        (panel_x + panel_w, panel_y + panel_h),
        (25, 25, 25),
        -1
    )

    frame = cv2.addWeighted(
        overlay,
        0.75,
        frame,
        0.25,
        0
    )

    # Yellow border
    cv2.rectangle(
        frame,
        (panel_x, panel_y),
        (panel_x + panel_w, panel_y + panel_h),
        (0, 215, 255),
        2
    )

    # Title
    cv2.putText(
        frame,
        "BOTTLE CAP INSPECTOR",
        (panel_x + 15, panel_y + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 215, 255),
        2
    )

    # Total
    cv2.putText(
        frame,
        f"Processed: {total}",
        (panel_x + 15, panel_y + 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    # Green indicator
    cv2.circle(
        frame,
        (panel_x + 25, panel_y + 88),
        9,
        (0, 255, 0),
        -1
    )

    cv2.putText(
        frame,
        f"Capped: {capped}",
        (panel_x + 45, panel_y + 94),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    # Red indicator
    cv2.circle(
        frame,
        (panel_x + 25, panel_y + 122),
        9,
        (0, 0, 255),
        -1
    )

    cv2.putText(
        frame,
        f"Uncapped: {uncapped}",
        (panel_x + 45, panel_y + 128),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    # ==========================================
    # ALERT
    # ==========================================

    if uncapped > 0:

        cv2.putText(
            frame,
            "WARNING: UNCAPPED BOTTLE DETECTED!",
            (30, height - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    else:

        cv2.putText(
            frame,
            "INSPECTION: OK",
            (30, height - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    # Save frame
    out.write(frame)

    # Display
    cv2.imshow(
        "Bottle Cap Inspector",
        frame
    )

    # Press Q to stop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
out.release()
cv2.destroyAllWindows()

print("Inspection completed.")
print(f"Output saved to: {OUTPUT_PATH}")
