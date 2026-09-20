# Title

How Many Missing Frames Before This Robot Misses?

# Description

Your video call can skip a few frames and still make sense. Could a robot
goalkeeper do the same?

We kept the authentic Microduck model, 15-shot suite, MuJoCo physics, official
walking policy, prediction gain, and strict whole-ball scoring frozen. Every
candidate observation came from rendered overhead-camera pixels. A deterministic
predeclared mask either delivered each packet immediately or discarded it, and
higher-loss conditions kept a strict subset of the frames kept below them.

With 115 of 210 packets actually missing, the goalkeeper still saved 15/15
shots. With 168 of 210 missing, it saved 10/15 and fell below our predeclared
12-save tolerance bar. At 192 of 210 missing, it saved 8/15. It never fell or
left the goal zone in any condition; it simply ran out of useful pictures.

This is a MuJoCo simulation result, not real-world proof. No delivery delay or
hidden ball position was added. The next test is burst loss plus a safe fallback.

Repository:
https://github.com/sarvob/sarvoday-microduck-lab

Challenge source: `challenges/015-external-camera-frame-loss/spec.json` and
`scripts/evaluate_goalkeeper_camera_frame_loss.py`. Reproducible result:
`artifacts/015-external-camera-frame-loss/result.json`.

Human goalkeeper hook: Pexels video 6084018 by Tima Miroshnichenko, used under
the Pexels license. Music was generated specifically for this episode.

#Robotics #ComputerVision #RobotGoalkeeper #Microduck #MuJoCo

# Keywords

robot goalkeeper, dropped frames, packet loss, computer vision, robot duck,
Microduck, MuJoCo simulation, robotics experiment, Wi-Fi packet loss,
embodied AI, robot perception, sensor reliability, loss detection
