# Cubify Anything

This repository includes the implementation of Cubify Transformer for CareMate for detecting 3D objects in an indoor scene environment and labeling via clip.

## Paper

**Apple**

[Cubify Anything: Scaling Indoor 3D Object Detection](https://arxiv.org/abs/2412.04458)

Justin Lazarow, David Griffiths, Gefen Kohavi, Francisco Crespo, Afshin Dehghan

## Usage

- Download the cutr_rgb.pth model into the root directory
- Add video from real world or simulation in the data folder. Each video frame should have a .wide folder containing the image file. 
- Run this command to save the mode output and label outputs:

  ```python
  python tools/demo_to_save.py ./data/simulation-xxx --model-path ./cutr_rgb.pth --device mps
  ```
  Set device to 'cuda' if not using mps.

  Additional arguments:

  --confidence.thres: Confidence threshold for CLIP to use while displaying labels
  
  --room_map_path The path to .json file containing the object names. CLIP uses these as labels. By default it is object-room_map.json
  
  --scoree-thresh: The threshold to use for Cubify Transformer in its predictions


  
  
