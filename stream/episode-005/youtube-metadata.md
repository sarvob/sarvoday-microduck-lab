# Title

Can This Tiny Robot Balance on a Moving Boat? | MuJoCo Robotics

# Description

Microduck learns to stay centered on a moving boat deck in MuJoCo. The controller passes all three 20-second harbor tests, then fails honestly in rough chop—revealing the next robotics problem to solve.

In this episode:
- the physical success gates behind the challenge
- why a frozen standing policy fails on a moving support surface
- how a small residual plus recenter controller reaches 3/3 harbor passes
- why rough chop still causes deck exits
- what command saturation and contact timing can teach us next

Code and reproducible results:
https://github.com/sarvob/sarvoday-microduck-lab

The official Microduck joint-level policies remain frozen. The experiment changes only a disclosed high-level residual and recenter controller. The surge profile remains held out because the training gate has not passed.

#Robotics #MuJoCo #ReinforcementLearning #RobotLearning #Simulation #OpenSourceRobotics #Microduck
