from transformers import pipeline
import numpy as np
import cv2
import insightface
from insightface.app import FaceAnalysis
from PIL import Image, ImageDraw


# Initialize face detection
app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# Initialize segmentation pipeline
segmenter = pipeline(model="mattmdjaga/segformer_b2_clothes")


def remove_face(img, mask):
    # Convert image to numpy array
    img_arr = np.asarray(img)
    
    # Run face detection
    faces = app.get(img_arr)
    
    # Get the first face
    faces = faces[0]['bbox']

    # Width and height of face
    w = faces[2] - faces[0]
    h = faces[3] - faces[1]

    # Make face locations bigger
    faces[0] = faces[0] - (w*0.5) # x left
    faces[2] = faces[2] + (w*0.5) # x right
    faces[1] = faces[1] - (h*0.5) # y top
    faces[3] = faces[3] + (h*0.2) # y bottom

    # Convert to [(x_left, y_top), (x_right, y_bottom)]
    face_locations = [(faces[0], faces[1]), (faces[2], faces[3])]

    # Draw black rect onto mask
    img1 = ImageDraw.Draw(mask)
    img1.rectangle(face_locations, fill=0)

    return mask

def segment_body(original_img, face=True):
    # Make a copy
    img = original_img.copy()
    
    # Segment image
    segments = segmenter(img)

    # Create list of masks
    segment_include = ["Hat", "Hair", "Sunglasses", "Upper-clothes", "Skirt", "Pants", "Dress", "Belt", "Left-shoe", "Right-shoe", "Face", "Left-leg", "Right-leg", "Left-arm", "Right-arm", "Bag","Scarf"]
    mask_list = []
    for s in segments:
        if(s['label'] in segment_include):
            mask_list.append(s['mask'])


    # Paste all masks on top of eachother 
    final_mask = np.array(mask_list[0])
    for mask in mask_list:
        current_mask = np.array(mask)
        final_mask = final_mask + current_mask
            
    # Convert final mask from np array to PIL image
    final_mask = Image.fromarray(final_mask)

    # Remove face
    if(face==False):
        final_mask = remove_face(img.convert('RGB'), final_mask)

    # Apply mask to original image
    img.putalpha(final_mask)

    return img, final_mask
