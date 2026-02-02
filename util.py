import numpy as np
import cv2

def NMS(boxes, class_ids, confidences, overlapThresh=0.5):
    """
    Applies Non-Maximum Suppression (NMS) to remove overlapping bounding boxes.
    Uses OpenCV's built-in fast implementation.
    """
    if len(boxes) == 0:
        return [], [], []

    # cv2.dnn.NMSBoxes expects boxes, scores, score_threshold, nms_threshold
    # We use 0.5 as the score threshold for validity here
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, overlapThresh)

    if len(indices) > 0:
        # indices usually comes as a tuple or list depending on version, flatten ensures consistency
        indices = indices.flatten()

        # Filter the original lists to keep only the best boxes
        return (
            [boxes[i] for i in indices],
            [class_ids[i] for i in indices],
            [confidences[i] for i in indices]
        )

    return [], [], []

def get_outputs(net):
    """
    Runs the forward pass on the neural network and returns raw detections.
    Includes a fix for different OpenCV versions (getUnconnectedOutLayers).
    """
    layer_names = net.getLayerNames()

    # --- COMPATIBILITY FIX ---
    # Newer OpenCV returns 1D array, Older returns 2D array. This handles both.
    try:
        out_layers = net.getUnconnectedOutLayers()
        if len(out_layers.shape) == 1:
            # OpenCV 4.6+ (1D array of indices, 0-based or 1-based depending on usage)
            # Usually needs explicit integer conversion
            output_layers = [layer_names[i - 1] for i in out_layers]
        else:
            # Older OpenCV (2D array)
            output_layers = [layer_names[i[0] - 1] for i in out_layers]
    except:
        # Fallback if standard attribute access fails (rare)
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    # Run the network
    outs = net.forward(output_layers)

    # Flatten the result and filter weak "objectness" confidence (< 0.1)
    # c[4] is the objectness score in YOLOv3
    outs = [c for out in outs for c in out if c[4] > 0.1]

    return outs

def draw(bbox, img):
    """
    Helper to draw the bounding box on an image (for debugging).
    """
    xc, yc, w, h = bbox

    # Calculate top-left and bottom-right coordinates
    x1 = int(xc - w / 2)
    y1 = int(yc - h / 2)
    x2 = int(xc + w / 2)
    y2 = int(yc + h / 2)

    img = cv2.rectangle(img,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0), 20)

    return img