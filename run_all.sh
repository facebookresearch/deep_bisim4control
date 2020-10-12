#!/bin/bash

NOW=$(date +"%m%d%H%M")

./$1 cartpole swingup 2 ${NOW}
./$1 reacher easy 2 ${NOW}
./$1 cheetah run 2 ${NOW}
./$1 finger spin 2 ${NOW}
# ./$1 ball_in_cup catch 2 ${NOW}
./$1 walker walk 2 ${NOW}
./$1 walker stand 2 ${NOW}
./$1 walker run 2 ${NOW}
# ./$1 acrobot swingup 2 ${NOW}
./$1 hopper stand 2 ${NOW}
./$1 hopper hop 2 ${NOW}
# ./$1 manipulator bring_ball 2 ${NOW}
# ./$1 humanoid stand 2 ${NOW}
# ./$1 humanoid walk 2 ${NOW} 
# ./$1 humanoid run 2 ${NOW}