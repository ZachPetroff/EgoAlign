# EgoAlign

</p>
<p align="center">
	<img src="examples/example4.png" alt="Shadow Viz">
</p>

A toolkit for aligning motion capture, egocentric video, and gaze data into a unified SE(3) coordinate system. Designed for multimodal sensor fusion, embodied perception, and analysis of real-world behavior.

Dependencies:
- ViT Pose
- SAM2
- Shadow.fileio

Shadow:

To detect the grounded foot at each time-step, run the following command:

`python detect_steps.py --shadow_dir <path/to/shadow/> --out_csv <detected_steps.csv>` 

The resulting .csv file will contain the grounded foot at each timestamp, calculated using Shadow's pressure sensors. The .csv also contains the 3D body positions.

</p>
<p align="center">
	<img height="300" min-width="100" src="examples/skeleton_viz.gif" alt="Shadow Viz">
</p>

DJI:

To track the subject throughout the third-person video and get the corresponding bounding boxes, run:

`python segment_video --frames <path/to/frames> --output <sam2_output.mp4> --fps <output_fps> --checkpoint <path/to/sam.pt> --config <path/to/config.yaml> --out_csv output_bboxes.csv`

Upon running, you will be prompted to select points belonging to the subject. After closing the window, the following window will appear so that you can verify the mask: 

</p>
<p align="center">
	<img src="examples/sam_mask.png" alt="SAM Mask">
</p>

Upon closing, the script automatically generate the segmentation masks and bounding boxes for the rest of the video.

To get the 2D pose of the subject in each frame, run the following command:

`python vitpose_inference.py --video <path/to/dji_video.mp4> --output_dir <path/to/vitpose_output> --bbox_csv <path/to/sam_bboxes.csv> --checkpoints_dir <path/to/vitpose_checkpoints>`

</p>
<p align="center">
	<img height="300" min-width="100" src="examples/vitpose.gif" alt="ViT Pose">
</p>

After getting the 2D pose keypoints and the grounded foot at each timestamp, find median step locations in DJI and Shadow space by running the following command:

`python process_heels.py `

To get the step positions in DJI space, this script extracts the 3D position of the grounded foot at each time-step by projecting the 2D heel position (using camera extrinsics/intrinsics) along a ray and finding where it intersects with the environment. It then filters out outlying points and finds the median heel position within each discrete step.

To get the step positions in Shadow space, foot-slippage artifacts are corrected, then the median heel position within each discrete step event is calculated. 

The script also time syncs the heel positions for later spatial alignment.



Aria: 

- Process semidense points

Align Shadow + DJI

- rigid_align.py to align heels with environment

Align Shadow + DJI + Aria 

- align_reconstructions.py to find cameras in DJI space
- find gaze intersections

View with walk_viewer.py
