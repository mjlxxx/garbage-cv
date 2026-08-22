"""
layout might not work
"""

import json
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms


with open("labels.json", "r") as f:
    class_names = json.load(f)

# color coding by category
CATEGORY_STYLE = {
    "cardboard": {"bin": "Recycling",              "color": (66, 135, 245)},   #orange
    "glass":     {"bin": "Recycling",              "color": (245, 194, 66)},   #blue
    "metal":     {"bin": "Recycling",              "color": (200, 200, 200)},  #silver
    "paper":     {"bin": "Recycling",              "color": (66, 245, 236)},   #yellow
    "plastic":   {"bin": "Recycling (check #)",    "color": (66, 245, 105)},   #green
    "trash":     {"bin": "Landfill",                "color": (80, 80, 80)},    #dark gray
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, len(class_names))
model.load_state_dict(torch.load("trash_classifier.pt", map_location=device))
model.to(device)
model.eval()

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
])


def draw_rounded_rect(img, top_left, bottom_right, color, thickness=-1, radius=15):
    """rectangle"""
    x1, y1 = top_left
    x2, y2 = bottom_right

    if thickness < 0:
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        for cx, cy in [(x1 + radius, y1 + radius), (x2 - radius, y1 + radius),
                       (x1 + radius, y2 - radius), (x2 - radius, y2 - radius)]:
            cv2.circle(img, (cx, cy), radius, color, -1)
    else:
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
        cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
        cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)


def overlay_panel(frame, predicted_class, confidence, style):
    """nicer looking bar"""
    h, w = frame.shape[:2]
    panel_w, panel_h = 340, 110
    x, y = 20, h - panel_h - 20  #bottom left

    overlay = frame.copy()
    draw_rounded_rect(overlay, (x, y), (x + panel_w, y + panel_h), (30, 30, 30), thickness=-1)
    frame[:] = cv2.addWeighted(overlay, 0.65, frame, 1.00, 0)

    cv2.putText(frame, predicted_class.upper(), (x + 20, y + 38),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, style["color"], 2, cv2.LINE_AA)

    cv2.putText(frame, f"-> {style['bin']}", (x + 20, y + 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    bar_x, bar_y, bar_w, bar_h = x + 20, y + 82, panel_w - 40, 14
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (70, 70, 70), -1)
    filled_w = int(bar_w * confidence)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled_w, bar_y + bar_h), style["color"], -1)
    cv2.putText(frame, f"{confidence*100:.0f}%", (bar_x + bar_w + 10, bar_y + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)



cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("cant open")
    exit()

print("open")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    box_size = int(h * 0.55)
    cx, cy = w // 2, h // 2
    x1, y1 = cx - box_size // 2, cy - box_size // 2
    x2, y2 = cx + box_size // 2, cy + box_size // 2

    roi = frame[y1:y2, x1:x2]
    rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    input_tensor = preprocess(rgb_roi).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top_prob, top_idx = torch.max(probabilities, dim=0)

    predicted_class = class_names[top_idx.item()]
    confidence = top_prob.item()
    style = CATEGORY_STYLE.get(predicted_class, {"bin": "Unknown", "color": (255, 255, 255)})

    cv2.rectangle(frame, (x1, y1), (x2, y2), style["color"], 3)
    cv2.putText(frame, "Place item here", (x1, y1 - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, style["color"], 2, cv2.LINE_AA)

    overlay_panel(frame, predicted_class, confidence, style)

    cv2.imshow("Trash Classifier - Live", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()