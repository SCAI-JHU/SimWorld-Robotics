# SimWorld-Robotics: Synthesizing Photorealistic and Dynamic Urban Environments for Multimodal Robot Navigation and Collaboration
<p align="center">
  <img src="images/overview.png" alt="overview" width="800">
</p>

**SimWorld-Robotics** is an extension of [SimWorld](https://github.com/SimWorld-AI/SimWorld) that introduces key features for embodied robotics research. These additions include procedural city generation, a traffic system, and support for a new embodied agent: the quadruped robot.

<!-- <div align="center">
    <a href="https://simworld-ai.github.io/"><img src="https://img.shields.io/badge/Website-SimWorld-blue" alt="Website" /></a>
</div> -->

## 🔥 News
 - 2025.10 The first formal release of **SimWorld-Robotics** has been published! 🚀
 - 2025.9 **SimWorld-Robotics** has been accepted to NeurIPS 2025 main track! 🎉


## 💡 Introduction
This repo serves as a benchmark platform for the SimWorld-MMNav and SimWorld-MRS in **SimWorld-Robotics**:

1.  A standardized OpenAI gym interface for connecting and evaluating various baselines.
2.  Procedural scene and task generation for creating diverse and scalable simulation environments and tasks.
3.  The SimWorld-20k dataset is available via this [link]().


### Project Structure
```bash
simworld_gym/ 
    baseline/           # Baselines used in SimWorld-MMNav and SimWorld-MRS
    SimWorldGym/
      config/           # Configuration files for assets and robots
      envs/             # Gym Environments for SimWorld-MMNav and SimWorld-MRS
      task_generator/   # Procedural task generation
      utils/            # Utility functions
readme.md
```

## Setup
Before installing **SimWorld-Robotics**, install the main library [SimWorld]((https://github.com/SimWorld-AI/SimWorld)) first.
```bash
git clone git@github.com:SCAI-JHU/CityGym.git
cd simworld_gym
cd SimWorldGym
pip install -e .
```