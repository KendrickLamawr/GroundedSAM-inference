import random
from dataclasses import dataclass
from typing import Any, List, Dict, Optional, Union, Tuple

import cv2
import torch
import requests
import numpy as np
from PIL import Image
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from transformers import AutoModelForMaskGeneration, AutoProcessor, pipeline, SamModel, SamProcessor
import os
import glob
import helper
#=====================================================================
device = 'cuda' if torch.cuda.is_available() else "cpu"
print(device)
images_dir = f"C:/Users/ongng/Downloads/fake-20250703T025750Z-1-001/fake/hard_non_main"
labels = ["the most salient vehicle.", "the most prominent vehicle."]
image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
threshold = .3
image_paths = []
detector_id = "IDEA-Research/grounding-dino-tiny"
segmenter_id = "facebook/sam-vit-base"

polygon_refinement= False
#=====================================================================
for extension in image_extensions:
    image_paths.extend(glob.glob(os.path.join(images_dir, extension)))
    # image_paths.extend(glob.glob(os.path.join(images_dir, extension.upper())))

print(f"Found {len(image_paths)} images in the target folder")

#=====================================================================
# detect and segment 
#=====================================================================
object_detector = pipeline(model=detector_id, task ="zero-shot-object-detection", device=device)
# segmentator = AutoModelForMaskGeneration.from_pretrained(segmenter_id).to(device)
# processor = AutoProcessor.from_pretrained(segmenter_id)
segmentator = SamModel.from_pretrained(segmenter_id).to(device)
processor = SamProcessor.from_pretrained(segmenter_id)
for image_url in image_paths:
    if isinstance(image_url, str):
        image = helper.load_image(image_url)
    else:
        image = image_url
    #detect
    detection_results= object_detector(image, candidate_labels=labels, threshold= threshold)
    detection_results= [helper.DetectionResult.from_dict(result) for result in detection_results]
    
    #segment
    boxes = helper.get_boxes(detection_results)
    inputs = processor(images =image, input_boxes=boxes, return_tensors="pt").to(device)
    outputs = segmentator(**inputs)
    masks= processor.post_process_masks(masks=outputs.pred_masks,
                                        original_sizes = inputs.original_sizes,
                                        reshaped_input_sizes = inputs.reshaped_input_sizes)[0]
    masks = helper.refine_masks(masks, polygon_refinement=polygon_refinement)
    for detection_result, mask in zip(detection_results, masks):
        detection_result.mask = mask
    # plot result
    helper.plot_detections(np.array(image), [detection_result], f"{image_url.split('.')[0]}_res.png")
    