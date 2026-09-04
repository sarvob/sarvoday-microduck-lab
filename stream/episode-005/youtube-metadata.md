# Title

Microduck Learns Its Sea Legs | Robot Balance Control in MuJoCo

# Description

What happens when a walking robot's floor starts moving?

We put Microduck on a pitching, rolling boat in MuJoCo and built a balance controller around frozen robot policies. It clears every 20-second harbor test, then rough water exposes exactly where the design breaks.

In this episode:
- the physical success gates behind a moving-support challenge
- why a frozen standing policy fails on a moving support surface
- how a small residual plus recenter controller reaches 3/3 harbor passes
- why rough chop still causes deck exits
- what command saturation and contact timing can teach us next

Code and reproducible results:
https://github.com/sarvob/sarvoday-microduck-lab

Technical disclosure: the official Microduck joint-level policies remain frozen. The experiment changes only a disclosed high-level residual and recenter controller. The water surface is visual; deterministic deck translation, roll, and pitch are the physical test inputs. The surge profile remains held out because the training gate has not passed.

#Robotics #MuJoCo #ReinforcementLearning #RobotLearning #Simulation #OpenSourceRobotics #Microduck
