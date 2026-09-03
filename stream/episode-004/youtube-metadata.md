# Title

We Taught Microduck When to Stop Rolling | Robot Control in MuJoCo

# Description

Twenty milliseconds separates a clean recovery from a failed controller
handoff. We search 26 transition timings for Microduck's official roulade and
stand policies, evaluate three perturbed starting poses, and explain why
controller transitions are a robotics problem of their own.

The final section gives a practical visual checklist for comparing the selected
0.82-second handoff with the first failing 0.84-second boundary case.

Code and measured results:
https://github.com/sarvob/sarvoday-microduck-lab/tree/main/artifacts/006-controlled-roll

The shipped Microduck joint-level policies remain frozen. The learned component
is the high-level roll-to-stand handoff time.

# Tags

robotics simulation, MuJoCo, Microduck robot, robot control, reinforcement
learning, humanoid robot, biped robot, robotics engineering, control policy,
simulator training, open source robotics, embodied AI
