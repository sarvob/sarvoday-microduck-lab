# Title

Can This Tiny Robot Duck Become a Goalkeeper? | Microduck Challenge

# Description

I wanted to know whether Microduck could become a goalkeeper. The problem was
that this tiny robot has no arms, cannot sidestep, and can lose sight of the ball
when it turns.

So the challenge was simple: keep the authentic Pollen Robotics walking policy,
give the duck only its head-camera view, and find a useful controller with a
small search that fits inside a couple of hours.

The reactive version saved 14 of 15 unseen shots. The version that predicted
where the ball would cross the goal line saved 15 of 15. No falls and no exits
from the goal zone.

Code and reproducible results:
https://github.com/sarvob/sarvoday-microduck-lab

Chapters:
00:00 Can a tiny duck keep goal?
00:08 The first attempt
00:40 Three things working against Microduck
01:31 The reactive goalkeeper
02:15 One small change
03:00 Six quick training options
03:45 The unseen final test
04:51 The result
05:03 Victory lap

Technical disclosure:

- Microduck, its head-camera images, ball motion, contacts, and scoring come
  from the measured MuJoCo experiment.
- The official Microduck walking network remains frozen.
- The controller detects the orange ball from camera pixels and estimates its
  goal-line crossing. It does not read hidden ball coordinates for decisions.
- The narrow right-side lab rail reports measurements but does not alter the
  simulation.
- The opening human-goalkeeper sequence is an original low-poly Blender
  animation, not celebrity or broadcast footage.
- Music and sound effects were generated specifically for this episode.

#Robotics #Microduck #MuJoCo #RobotLearning #ComputerVision

# Thumbnail text

CAN THIS DUCK SAVE THE GOAL?
