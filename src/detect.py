import torch
import cv2

model = torch.hub.load('./src/yolov7', 'custom', './herramientas/models/modelo-detector-ArgentinIA.pt', source = "local",
                        force_reload=True, trust_repo=True)

def detect(image):
    results = model(image)
    df = results.pandas().xyxy[0]
    return df
