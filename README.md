# depth_avoider_crazyflie
Supplement materials for paper: "Nano Quadcopter Obstacle Avoidance with a Lightweight Monocular Depth Network" submitted to IFAC World Congress 2023.

## Experiment Video
https://youtu.be/ss8BPzg_JyY

The above video is attached to give reviewers a clearer view of 1. flight trajectories and 2. real-time onboard camera images with predicted depth maps in the paper.

The included experiments:
- With depth network trained in the CyberZoo environment:
  1. Evaluated in the CyberZoo *sparse / dense* environments with fixed obstacles;
  2. Transferred to the CyberZoo environment with *dynamic / unseen* obstacles;
  3. Transferred to the Corridor environment.
- With depth network trained & evaluated in the Corridor environment.

## Source Code
- `NanoDepth.py` contains the nano depth convolutional neural network framework written in PyTorch.
- `bsm.py` contains the behavior state machine based on depth map for obstacle avoidance.
