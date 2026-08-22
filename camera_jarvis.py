

import json
import math
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from ultralytics import YOLO


CYAN   = (255, 255, 0)    
AMBER  = (0, 170, 255)    
WHITE  = (255, 255, 255)
DARK   = (25, 20, 15)      

CATEGORY_STYLE = {
    "cardboard": {"bin": "RECYCLE",  "color": (66, 135, 245)},
    "glass":     {"bin": "RECYCLE",  "color": (245, 194, 66)},
    "metal":     {"bin": "RECYCLE",  "color": (200, 200, 200)},
    "paper":     {"bin": "RECYCLE",  "color": (66, 245, 236)},
    "plastic":   {"bin": "RECYCLE*", "color": (66, 245, 105)},
    "trash":     {"bin": "LANDFILL", "color": (100, 100, 255)},
}


with open("labels.json", "r") as f:
    class_names = json.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trash_model = models.mobilenet_v2(weights=None)
trash_model.classifier[1] = nn.Linear(trash_model.last_channel, len(class_names))
trash_model.load_state_dict(torch.load("trash_classifier.pt", map_location=device))
trash_model.to(device)
trash_model.eval()

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
])

yolo = YOLO("yolov8n.pt")

MIN_BOX_SIZE = 40
YOLO_CONF_THRESHOLD = 0.35
ALLOWED_YOLO_CLASSES = {
    "bottle", "cup", "wine glass", "bowl", "banana", "apple", "orange",
    "sandwich", "book", "vase", "scissors", "toothbrush", "fork", "knife",
    "spoon", "remote", "cell phone", "handbag", "backpack", "suitcase",
    "frisbee", "sports ball", "teddy bear", "hair drier",
}


def pad_to_square(img, pad_color=(0, 0, 0)):
    h, w = img.shape[:2]
    size = max(h, w)
    top, bottom = (size - h) // 2, size - h - (size - h) // 2
    left, right = (size - w) // 2, size - w - (size - w) // 2
    return cv2.copyMakeBorder(img, top, bottom, left, right,
                               cv2.BORDER_CONSTANT, value=pad_color)


def classify_crop(crop_bgr):
    crop_bgr = pad_to_square(crop_bgr)
    rgb_crop = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    input_tensor = preprocess(rgb_crop).unsqueeze(0).to(device)
    with torch.no_grad():
        output = trash_model(input_tensor)
        probs = torch.nn.functional.softmax(output[0], dim=0)
        top_prob, top_idx = torch.max(probs, dim=0)
    return class_names[top_idx.item()], top_prob.item()



def pulse(frame_count, speed=0.12, low=140, high=255):
    """Smooth oscillating brightness value for a 'breathing glow' effect."""
    return int(low + (high - low) * (0.5 + 0.5 * math.sin(frame_count * speed)))


def draw_corner_brackets(frame, x1, y1, x2, y2, color, length=22, thickness=2):
    """Sci-fi targeting reticle: only the corners are drawn, not a full box."""
    for (cx, cy, dx, dy) in [(x1, y1, 1, 1), (x2, y1, -1, 1),
                              (x1, y2, 1, -1), (x2, y2, -1, -1)]:
        cv2.line(frame, (cx, cy), (cx + dx * length, cy), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (cx, cy), (cx, cy + dy * length), color, thickness, cv2.LINE_AA)


def draw_grid_overlay(frame, spacing=60, color=CYAN, alpha=0.06):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    for x in range(0, w, spacing):
        cv2.line(overlay, (x, 0), (x, h), color, 1, cv2.LINE_AA)
    for y in range(0, h, spacing):
        cv2.line(overlay, (0, y), (w, y), color, 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def draw_scanline(frame, frame_count, color=CYAN, period=140):
    h, w = frame.shape[:2]
    pos = int(abs((frame_count % (2 * period)) - period) / period * h)
    overlay = frame.copy()
    cv2.line(overlay, (0, pos), (w, pos), color, 2, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)


def draw_radar(frame, center, radius, angle_deg, color=CYAN):
    overlay = frame.copy()
    cv2.circle(overlay, center, radius, color, 1, cv2.LINE_AA)
    cv2.circle(overlay, center, radius // 2, color, 1, cv2.LINE_AA)
    x2 = int(center[0] + radius * math.cos(math.radians(angle_deg)))
    y2 = int(center[1] + radius * math.sin(math.radians(angle_deg)))
    cv2.line(overlay, center, (x2, y2), color, 2, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)


def draw_top_bar(frame, item_count, frame_count):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 50), DARK, -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.line(frame, (0, 50), (w, 50), CYAN, 1, cv2.LINE_AA)

    glow = pulse(frame_count)
    title_color = (glow, glow, 0)
    cv2.putText(frame, "W.A.S.T.E. VISION SYSTEM", (15, 32),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, title_color, 1, cv2.LINE_AA)

    status_dot_color = (0, 255, 0) if item_count > 0 else (0, 100, 200)
    cv2.circle(frame, (w - 130, 25), 6, status_dot_color, -1, cv2.LINE_AA)
    status_text = "TRACKING" if item_count > 0 else "SCANNING"
    cv2.putText(frame, status_text, (w - 115, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)


def draw_item_tag(frame, x1, y1, x2, y2, predicted_class, confidence, style, frame_count):
    glow = pulse(frame_count)
    color = tuple(min(255, int(c * glow / 255) + 40) for c in style["color"])

    draw_corner_brackets(frame, x1, y1, x2, y2, color)

    label_y = max(60, y1 - 30)
    cv2.line(frame, (x1, y1), (x1, label_y + 8), color, 1, cv2.LINE_AA)

    text_main = f"{predicted_class.upper()}"
    text_sub = f"CONF {confidence*100:0.1f}%  |  {style['bin']}"

    (tw, th), _ = cv2.getTextSize(text_main, cv2.FONT_HERSHEY_DUPLEX, 0.55, 1)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1 - 4, label_y - th - 22), (x1 + tw + 90, label_y + 6), DARK, -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, text_main, (x1, label_y - 10),
                cv2.FONT_HERSHEY_DUPLEX, 0.55, color, 1, cv2.LINE_AA)
    cv2.putText(frame, text_sub, (x1, label_y + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, WHITE, 1, cv2.LINE_AA)


def draw_bottom_panel(frame, item_count, fps):
    h, w = frame.shape[:2]
    panel_h = 34
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - panel_h), (w, h), DARK, -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.line(frame, (0, h - panel_h), (w, h - panel_h), CYAN, 1, cv2.LINE_AA)

    cv2.putText(frame, f"OBJECTS: {item_count}", (15, h - 11),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, CYAN, 1, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:0.1f}", (w - 110, h - 11),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, CYAN, 1, cv2.LINE_AA)



cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not open webcam. Check camera permissions in System Settings > Privacy & Security > Camera.")
    exit()

print("HUD initializing. Press 'q' to quit.")

frame_count = 0
radar_angle = 0
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    now = time.time()
    fps = 1.0 / max(now - prev_time, 1e-6)
    prev_time = now

    results = yolo(frame, verbose=False)[0]

    item_count = 0
    for box in results.boxes:
        conf = float(box.conf[0])
        if conf < YOLO_CONF_THRESHOLD:
            continue

        yolo_class_name = yolo.names[int(box.cls[0])]
        if yolo_class_name not in ALLOWED_YOLO_CLASSES:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        if (x2 - x1) < MIN_BOX_SIZE or (y2 - y1) < MIN_BOX_SIZE:
            continue

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        predicted_class, trash_conf = classify_crop(crop)
        style = CATEGORY_STYLE.get(predicted_class, {"bin": "UNKNOWN", "color": WHITE})

        draw_item_tag(frame, x1, y1, x2, y2, predicted_class, trash_conf, style, frame_count)
        item_count += 1

    # HUD chrome (drawn every frame regardless of detections)
    draw_grid_overlay(frame)
    draw_scanline(frame, frame_count)
    draw_radar(frame, (w - 55, h - 90), 35, radar_angle)
    draw_top_bar(frame, item_count, frame_count)
    draw_bottom_panel(frame, item_count, fps)

    cv2.imshow("W.A.S.T.E. Vision System", frame)

    frame_count += 1
    radar_angle = (radar_angle + 4) % 360

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()