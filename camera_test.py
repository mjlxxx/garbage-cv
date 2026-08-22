"""
base camera + CV test
opens webcam runs each frame thoruhg  pretrained image classifier
display live label on screen, q to quit
"""

import cv2
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models import MobileNet_V2_Weights

#pretrained model laoding
weights = MobileNet_V2_Weights.DEFAULT
model = models.mobilenet_v2(weights=weights)
model.eval()

#human labels??
labels = weights.meta["categories"]

#pipeline for preprocessing the image to fit the model
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
])

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("cant open")
    exit()

print("open, q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    #convert bgr format to rgb
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    input_tensor = preprocess(rgb_frame)
    input_batch = input_tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(input_batch)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top_prob, top_idx = torch.max(probabilities, dim=0)

    label = labels[top_idx.item()]
    confidence = top_prob.item()

    #prediciton text oevrlay, fix this
    text = f"{label}: {confidence*100:.1f}%"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Garbage CV - Base Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()