# YOLO Object Detection and the License Plate Project

> Study guide. Read it, run `example.py`, then write the topic **in your own words** and put photos of your handwritten pages in `my_handwritten_notes/`.

## What is object detection?
Finding **what** objects are in an image and **where**: for each object the model outputs a **bounding box**, a **class** and a **confidence** score. (Classification only says what is in the image; detection also localizes.)

## YOLO: "You Only Look Once"
A **single-stage** detector: one forward pass of one neural network predicts all boxes and classes at once, which makes it **fast enough for real-time video**. (Two-stage detectors like Faster R-CNN propose regions first, then classify them: usually slower.)

Each prediction contains: box position (x, y), size (width, height), an **objectness/confidence** score and **class probabilities**.

## Key concepts
- **IoU (Intersection over Union)** = area of overlap / area of union of two boxes. Measures how well a predicted box matches the true one (1 = perfect).
- **Confidence threshold**: discard boxes below this score (we used about 0.35).
- **Non-Maximum Suppression (NMS)**: many boxes land on the same object; keep the highest-confidence one and remove others with high IoU overlap.
- **Anchors / grid**: older YOLO versions divide the image into a grid and each cell predicts boxes; newer versions (YOLOv8) are anchor-free.

## Training metrics
**Precision**, **recall**, **mAP50** (mean average precision counting a detection correct when IoU >= 0.5) and **mAP50-95** (average over IoU 0.5 to 0.95, stricter). Also box loss, class loss.

## Label format (YOLO)
One .txt per image, one line per object: `class x_center y_center width height`, all **normalized to 0-1**.

## Transfer learning
We start from pretrained weights (`yolov8n.pt`, already trained on the COCO dataset) and **fine-tune** on our license plate images. This needs far less data and time than training from scratch.

## Our license plate project (pipeline)
Video or webcam -> YOLO detects the plate -> crop the plate region -> OCR (EasyOCR) reads the text -> save to CSV.
- A simple **tracker** matches plates across frames by IoU so each plate is saved **once**.
- Trained on the Kaggle "Car License Plate Detection" dataset in Google Colab (T4 GPU); the web app is built with Streamlit.
- Project repository: https://github.com/bigesh11/license-plate-monitor
