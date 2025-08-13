import os
import glob
import torch
import numpy as np
import cv2
import json

from torch.utils.data import IterableDataset

from cubifyanything.sensor import SensorArrayInfo, PosedSensorInfo
from cubifyanything.measurement import ImageMeasurementInfo, DepthMeasurementInfo
from cubifyanything.instances import Instances3D
from cubifyanything.boxes import GeneralInstance3DBoxes

def create_dummy_sample():
    """
    Creates a dummy 'sample' variable with the expected structure for the model.
    """
    # 1. Meta information
    meta = {"video_id": 999, "timestamp": 123456789.0}

    # 2. Sensor Information (Intrinsics, Poses, Sizes)
    image_size = (640, 480)  # (width, height)
    depth_size = (320, 240)  # (width, height)

    # Dummy camera intrinsics matrix K (with batch dimension)
    image_K = torch.tensor([[
        [500.0, 0.0, image_size[0] / 2],
        [0.0, 500.0, image_size[1] / 2],
        [0.0, 0.0, 1.0]
    ]])
    depth_K = torch.tensor([[
        [250.0, 0.0, depth_size[0] / 2],
        [0.0, 250.0, depth_size[1] / 2],
        [0.0, 0.0, 1.0]
    ]])

    # Dummy pose matrix RT (identity, with batch dimension)
    pose_RT = torch.eye(4).unsqueeze(0)

    # Create MeasurementInfo objects
    image_info = ImageMeasurementInfo(size=image_size, K=image_K)
    depth_info = DepthMeasurementInfo(size=depth_size, K=depth_K)

    # Create PosedSensorInfo objects
    wide_sensor_info = PosedSensorInfo()
    wide_sensor_info.image = image_info
    wide_sensor_info.depth = depth_info
    wide_sensor_info.RT = pose_RT

    gt_sensor_info = PosedSensorInfo()
    gt_sensor_info.depth = depth_info
    gt_sensor_info.RT = pose_RT

    # Create the main SensorArrayInfo
    sensor_info = SensorArrayInfo()
    sensor_info.wide = wide_sensor_info
    sensor_info.gt = gt_sensor_info

    # 3. Actual Sensor Data Tensors
    # Dummy image (batch, channels, height, width)
    dummy_image = torch.rand(1, 3, image_size[1], image_size[0])
    # Dummy depth map (batch, channels, height, width)
    dummy_depth = torch.rand(1, 1, depth_size[1], depth_size[0])
    dummy_gt_depth = torch.rand(1, 1, depth_size[1], depth_size[0])

    # 4. Instances
    # Create an empty Instances3D object
    instances = Instances3D()
    instances.set("gt_ids", [])
    instances.set("gt_names", [])
    # The model expects GeneralInstance3DBoxes for gt_boxes_3d
    empty_boxes = GeneralInstance3DBoxes(
        np.empty((0, 6), dtype=np.float32),
        np.empty((0, 3, 3), dtype=np.float32)
    )
    instances.set("gt_boxes_3d", empty_boxes)

    # 5. Assemble the final sample dictionary
    sample = {
        "meta": meta,
        "sensor_info": sensor_info,
        "wide": {
            "image": dummy_image,
            "depth": dummy_depth,
            "instances": instances,
        },
        "gt": {
            "depth": dummy_gt_depth,
        }
    }
    return sample

def preprocess_video(rgb_video_path, depth_video_path, save_path, target_fps=20, max_width=1024):
    """
    Preprocesses the RGB and depth videos to create a dataset.
    """
    cap_rgb = cv2.VideoCapture(rgb_video_path)
    cap_depth = cv2.VideoCapture(depth_video_path)

    if not cap_rgb.isOpened() or not cap_depth.isOpened():
        raise IOError("Cannot open one of the video files.")
    video_fps = cap_rgb.get(cv2.CAP_PROP_FPS)
    frame_skip = int(round(video_fps / target_fps))
    frame_idx = 0
    while True:
        ret_rgb, frame_rgb = cap_rgb.read()
        ret_depth, frame_depth = cap_depth.read()

        if not ret_rgb or not ret_depth:
            break

        if frame_idx % frame_skip == 0:
            # Convert RGB frame from BGR to RGB
            frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB)
            if frame_rgb.shape[1] > max_width:
                scale_factor = max_width / frame_rgb.shape[1]
                new_size = (max_width, int(frame_rgb.shape[0] * scale_factor))
                frame_rgb = cv2.resize(frame_rgb, new_size, interpolation=cv2.INTER_AREA)
                frame_depth = cv2.resize(frame_depth, new_size, interpolation=cv2.INTER_AREA)
            # Save or process the frames as needed
            os.makedirs(f"{save_path}_{frame_idx}", exist_ok=True)
            cv2.imwrite(f"{save_path}_{frame_idx}/rgb.png", frame_rgb)
            cv2.imwrite(f"{save_path}_{frame_idx}/depth.png", frame_depth)
            print(f"Processed frame {frame_idx}")
        frame_idx += 1

class CustomizeVideoDataset(IterableDataset):
    def __init__(self, data_path, max_width=1024):
        super().__init__()
        self.data_path = data_path
        self.max_width = max_width
        self.frame_paths = sorted(glob.glob(f"{data_path}/*.wide"))
        print(f"Found {len(self.frame_paths)} frames in {data_path}")

        # sample_rgb = cv2.imread(os.path.join(self.frame_paths[0], "image.png"))
        # self.frame_size = sample_rgb.shape[:2]

    def resize(self, img, h_scale, w_scale):
        """
        Resize the image tensor to the specified height and width scales.
        """
        new_size = (int(img.shape[1] * w_scale), int(img.shape[0] * h_scale))
        return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)

    def __iter__(self):
        for frame_path in self.frame_paths:
            frame_id = frame_path.split('/')[-1].split('.')[0]
            

            rgb_frame = cv2.imread(os.path.join(frame_path, "image.png"))
            depth_frame = cv2.imread(os.path.join(frame_path, "depth.png"))

            if rgb_frame is None or depth_frame is None:
                raise ValueError(f"Could not read frames from {frame_path}")
            
            # Ensure the RGB frame is resized to max_width if necessary
            if rgb_frame.shape[1] > self.max_width:
                scale_factor = self.max_width / rgb_frame.shape[1]
                rgb_frame = self.resize(rgb_frame, scale_factor, scale_factor)
                depth_frame = self.resize(depth_frame, scale_factor, scale_factor)
            # decrease resolution by half for depth
            # depth_frame = self.resize(depth_frame, 0.5, 0.5)
            # convert to grayscale
            depth_frame = cv2.cvtColor(depth_frame, cv2.COLOR_BGR2GRAY)

            # Create sensor info
            RT_matrix = torch.tensor(
                json.load(open(f"{self.data_path}/{frame_id}.gt/RT.json"))
            ).reshape(4, 4).float().unsqueeze(0)  # Convert to tensor and add batch dimension
            K_matrix = torch.tensor(
                json.load(open(f"{self.data_path}/{frame_id}.gt/K.json"))
            ).reshape(3, 3).float().unsqueeze(0)
            wide = PosedSensorInfo(
                image=ImageMeasurementInfo(size=rgb_frame.shape[:2], K=K_matrix),
                depth=DepthMeasurementInfo(size=depth_frame.shape[:2], K=K_matrix),
                # RT=RT_matrix
                RT=torch.eye(4).unsqueeze(0)  # Identity for wide
            )
            gt = PosedSensorInfo(
                depth=DepthMeasurementInfo(size=depth_frame.shape[:2], K=K_matrix),
                # RT=RT_matrix
                RT=torch.eye(4).unsqueeze(0)  # Identity for GT
            )
            sensor_info = SensorArrayInfo(wide=wide, gt=gt)

            yield {
                "meta": {
                    "video_id": "simulation01",
                    "timestamp": float(frame_id)
                }, 
                "sensor_info": sensor_info,
                "wide": {
                    "image": torch.tensor(rgb_frame).int().permute(2, 0, 1).unsqueeze(0),  # Convert to (B, C, H, W)
                    "depth": torch.tensor(depth_frame).int().unsqueeze(0),  # Convert to (B, H, W)
                    "instances": Instances3D(),
                },
                "gt": {
                    "depth": torch.tensor(depth_frame).int().unsqueeze(0)
                }
            }

if __name__ == "__main__":
    # Example usage
    rgb_video_path = "./data/simulation/rgb.mp4"
    depth_video_path = "./data/simulation/depth_clip.mp4"
    save_path = "./data/simulation/frames"

    preprocess_video(rgb_video_path, depth_video_path, save_path, target_fps=1)

    ds = CustomizeVideoDataset(data_path="./data/simulation/")
    sample = next(iter(ds))
    print(sample['wide']['image'].shape)  # Should print the shape of the RGB image tensor
    print(sample['wide']['depth'].shape)  # Should print the shape of the depth image tensor
    import matplotlib.pyplot as plt
    plt.imshow(sample['wide']['image'].squeeze(0).permute(1, 2, 0).numpy())
    plt.title("RGB Image")
    plt.axis('off')
    plt.show()
    plt.imshow(sample['wide']['depth'].squeeze(0).numpy(), cmap='gray')
    plt.title("Depth Image")
    plt.axis('off') 
    plt.show()
    
    # You can now create and inspect the dummy sample:
    # sample = create_dummy_sample()
    # print(sample)
