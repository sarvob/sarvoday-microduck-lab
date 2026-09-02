# YouTube metadata — Episode 003 package

## Main episode

**Title:** Can a Robot Duck Land on a Robot Dog? MuJoCo Training

**Description:**

We trained a Pollen Microduck landing controller to reach a Unitree Go1 in
MuJoCo, land with both feet, and remain stable across three randomized tests.

The honest constraint: the stock XL330-class behavior could not produce the
required unassisted height. We measured only 0.0737 m of rise and 0.02 seconds
of true airtime. The final experiment therefore uses one disclosed initial
launch velocity—1.2 m/s forward and 3.3 m/s vertical—with zero external force
in flight. The learned component is the randomized landing and stable hold.

Results: 1.68 s, 1.64 s, and 1.64 s holds across seeds 17, 71, and 173, with
both feet on the platform and no later ground contact.

Code and measured evidence:
https://github.com/sarvob/sarvoday-microduck-lab/tree/main/artifacts/005-duck-quadruped-jump

Microduck project: https://github.com/pollen-robotics/microduck

Unitree Go1 model: https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_go1

**Keywords:** robot learning, reinforcement learning, MuJoCo, Microduck,
Unitree Go1, robotics simulation, AI robotics, biped robot, quadruped robot,
robot landing control

## Short A

**Title:** We Refused to Fake This Robot Jump

**Description:** The stock Microduck policy rose only 0.0737 m—far below the
0.417 m target. Here is why we disclosed a one-time launch assist and trained
only the landing. Code and data: https://github.com/sarvob/sarvoday-microduck-lab

## Short B

**Title:** Robot Duck Lands on a Unitree Go1 — 3/3 Tests

**Description:** Both feet down, no later ground contact, and stable holds of
1.68, 1.64, and 1.64 seconds across three randomized MuJoCo tests. Code and
evidence: https://github.com/sarvob/sarvoday-microduck-lab

## Suggested hashtags

#Robotics #RobotLearning #MuJoCo #ReinforcementLearning #Unitree #Microduck
