# Title

Can a Robot Goalkeeper Survive Four Seconds of Lag?

# Description

Microduck's overhead camera saved 15 of 15 shots when its rendered pixels
arrived immediately. But what happens when those same frames arrive late?

We kept the authentic robot model, 15-shot suite, MuJoCo physics, official
walking policy, prediction gain, and strict whole-ball scoring frozen. Each
camera observation retained its real capture timestamp, but the controller
could not see it until the declared delivery time.

The goalkeeper saved 15/15 shots at zero, one, and two seconds of delay. At
three seconds it saved 12/15, exactly meeting the predeclared tolerance gate.
At four seconds it saved only 7/15. It never fell or left the goal zone; the
control loop simply received useful information too late.

This is a MuJoCo simulation result, not real-world proof. No frames were dropped
in this experiment; dropped-frame tolerance is the next test.

Repository:
https://github.com/sarvob/sarvoday-microduck-lab

Challenge source: `challenges/014-external-camera-latency/spec.json` and
`scripts/evaluate_goalkeeper_camera_latency.py`. Reproducible result:
`artifacts/014-external-camera-latency/result.json`.

Human goalkeeper hook: Pexels video 6084018 by Tima Miroshnichenko, used under
the Pexels license. Music was generated specifically for this episode.

#Robotics #ComputerVision #RobotGoalkeeper #Microduck #MuJoCo

# Keywords

robot goalkeeper, camera latency, computer vision latency, robot duck,
Microduck, MuJoCo simulation, robotics experiment, control loop, embodied AI,
robot perception, sensor delay, robot learning
