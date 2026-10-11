"""IoU and Non-Maximum Suppression, the two ideas behind every YOLO detector."""


def iou(a, b):
    """Boxes are (x1, y1, x2, y2)."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def nms(detections, iou_threshold=0.5):
    """detections = [(box, confidence)]. Keep the best box, drop heavy overlaps, repeat."""
    remaining = sorted(detections, key=lambda d: d[1], reverse=True)
    kept = []
    while remaining:
        best = remaining.pop(0)
        kept.append(best)
        remaining = [d for d in remaining if iou(best[0], d[0]) < iou_threshold]
    return kept


print("IoU of identical boxes:      ", iou((0, 0, 10, 10), (0, 0, 10, 10)))
print("IoU of half-overlapping boxes:", round(iou((0, 0, 10, 10), (5, 0, 15, 10)), 3), "(= 50 / 150)")
print("IoU of separate boxes:       ", iou((0, 0, 10, 10), (20, 20, 30, 30)))

# Three boxes on the same plate and one on a different plate
boxes = [
    ((100, 100, 200, 140), 0.91),
    ((105, 102, 205, 143), 0.84),    # duplicate of the first
    ((98, 98, 198, 138), 0.60),      # another duplicate
    ((300, 200, 400, 240), 0.77),    # a different plate
]
print("\nBefore NMS:", len(boxes), "boxes")
for box, conf in nms(boxes):
    print("After NMS keep:", box, "confidence", conf)
