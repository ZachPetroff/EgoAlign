# EgoAlign

![](examples/example4.png)

A toolkit for aligning motion capture, egocentric video, and gaze data into a unified SE(3) coordinate system. Designed for multimodal sensor fusion, embodied perception, and analysis of real-world behavior.

Dependencies:
- ViT Pose
- SAM2
- Shadow.fileio

Shadow:

To detect the grounded foot at each time-step, run the following command:

`python detect_steps.py --shadow_dir <path/to/shadow/> --out_csv <detected_steps.csv>` 

The resulting .csv file will contain the grounded foot at each timestamp, calculated using Shadow's pressure sensors. The .csv also contains the 3D body positions.

  

DJI:

- Run SAM2
- Run ViT
- Filter ViT using SAM2
- Find heel intersections

Aria: 

- Process semidense points

Align Shadow + DJI

- rigid_align.py to align heels with environment

Align Shadow + DJI + Aria 

- align_reconstructions.py to find cameras in DJI space
- find gaze intersections

View with walk_viewer.py
