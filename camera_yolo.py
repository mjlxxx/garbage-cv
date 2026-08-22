
import json
import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from ultralytics import YOLO

#setup
with open("labels.json", "r") as f:
    class_names = json.load(f)

CATEGORY_STYLE = {
    "cardboard": {"bin": "Recycling",           "color": (66, 135, 245)},
    "glass":     {"bin": "Recycling",           "color": (245, 194, 66)},
    "metal":     {"bin": "Recycling",           "color": (200, 200, 200)},
    "paper":     {"bin": "Recycling",           "color": (66, 245, 236)},
    "plastic":   {"bin": "Recycling (check #)", "color": (66, 245, 105)},
    "trash":     {"bin": "Landfill",            "color": (80, 80, 80)},
}

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

#for object localizer
yolo = YOLO("yolov8n.pt") 

MIN_BOX_SIZE = 40        #ignore small
YOLO_CONF_THRESHOLD = 0.35

#only specific classes so person not detected adjust color rating
ALLOWED_YOLO_CLASSES = {
    "bottle", "cup", "wine glass", "bowl", "banana", "apple", "orange",
    "sandwich", "book", "vase", "scissors", "toothbrush", "fork", "knife",
    "spoon", "remote", "cell phone", "handbag", "backpack", "suitcase",
    "frisbee", "sports ball", "teddy bear", "hair drier",
}


def pad_to_square(img, pad_color=(0, 0, 0)):
    """preserving padding"""
    h, w = img.shape[:2]
    size = max(h, w)
    top = (size - h) // 2
    bottom = size - h - top
    left = (size - w) // 2
    right = size - w - left
    return cv2.copyMakeBorder(img, top, bottom, left, right,
                               cv2.BORDER_CONSTANT, value=pad_color)


def classify_crop(crop_bgr):
    """run region again through old one"""
    crop_bgr = pad_to_square(crop_bgr)
    rgb_crop = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    input_tensor = preprocess(rgb_crop).unsqueeze(0).to(device)
    with torch.no_grad():
        output = trash_model(input_tensor)
        probs = torch.nn.functional.softmax(output[0], dim=0)
        top_prob, top_idx = torch.max(probs, dim=0)
    return class_names[top_idx.item()], top_prob.item()


def draw_label(frame, x1, y1, x2, y2, text, color):
    """bound box and tag"""
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    tag_y1 = max(0, y1 - text_h - 12)
    cv2.rectangle(frame, (x1, tag_y1), (x1 + text_w + 12, y1), color, -1)
    cv2.putText(frame, text, (x1 + 6, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


# ---------- Main loop ----------

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("cant open camera")
    exit()

print("open")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    #localization boxes
    results = yolo(frame, verbose=False)[0]

    item_count = 0
    for box in results.boxes:
        conf = float(box.conf[0])
        if conf < YOLO_CONF_THRESHOLD:
            continue

        yolo_class_name = yolo.names[int(box.cls[0])]
        if yolo_class_name not in ALLOWED_YOLO_CLASSES:
            continue  #skips person and stuff

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        if (x2 - x1) < MIN_BOX_SIZE or (y2 - y1) < MIN_BOX_SIZE:
            continue

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        predicted_class, trash_conf = classify_crop(crop)
        style = CATEGORY_STYLE.get(predicted_class, {"bin": "Unknown", "color": (255, 255, 255)})

        label = f"{predicted_class} {trash_conf*100:.0f}% -> {style['bin']}"
        draw_label(frame, x1, y1, x2, y2, label, style["color"])
        item_count += 1

    cv2.putText(frame, f"Items detected: {item_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("Trash Detector - Multi-Item", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()