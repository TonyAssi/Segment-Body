# Segment Body
Segment body, with or without face.

Takes in a PIL image and outputs the segmented body and mask. Built on top of 🤗 Tranformers using the [mattmdjaga/segformer_b2_clothes](https://huggingface.co/mattmdjaga/segformer_b2_clothes) image segmentation model.

## Installation
```bash
pip install -r requirements.txt
```

## Usage

Import module
```python
from SegBody import segment_body
```

Import PIL and open image
```python
from PIL import Image
image = Image.open('image.jpg')
```
---

Segment body with face
- **img** input image of type PIL
```python
result = segment_body(img=image)
result.save('segmented.png')
```

Segment body **without** face
- **img** input image of type PIL
- **face** Boolean, True by default
```python
result = segment_clothing(img=image, face=False)
result.save('segmented.png')
```
