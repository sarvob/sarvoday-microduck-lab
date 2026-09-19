# Title

Can a Second Camera Save This Robot Duck?

# Description

Microduck's head camera was completely blocked by a visual-only screen. With
the robot, shot generator, physics, prediction gain, and official walking policy
kept frozen, the head-camera condition saved only 2 of 15 unseen shots.

Then we added one calibrated overhead camera. Its rendered pixels—not hidden
simulator coordinates—supplied the same trajectory predictor. The outside-camera
condition saved 15 of 15 unseen shots, with no falls or goal-zone exits.

This is a MuJoCo simulation result, not real-world proof. The screen and outside
camera cannot touch the robot or ball. Strict scoring requires the complete ball
to remain out of the goal after contact.

Code and reproducible evidence:
https://github.com/sarvob/sarvoday-microduck-lab/tree/main/artifacts/013-external-camera-goalkeeper

Human goalkeeper hook: Pexels video 6084018 by Tima Miroshnichenko, used under
the Pexels license. Music was generated specifically for this episode.

#Robotics #ComputerVision #Microduck #MuJoCo #RobotLearning

# Keywords

robot goalkeeper, computer vision, overhead camera, robot duck, Microduck,
MuJoCo simulation, robotics experiment, occlusion, embodied AI, robot learning
