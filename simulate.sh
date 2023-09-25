#!/bin/bash

#iterations
n=1000
#players
p1="0"
p2="0"
p3="0"
scores=(0 0 0)
variance=(0 0 0)

for ((i=0 ; i < n ; i++)); do
	printf "$p1\n$p2\n$p3\n" | bash clock_game.sh True $RANDOM &&
		sleep 1
	sed -n '26,28 p' log_moves.txt | awk '{print $5}' > tmp
	mapfile -t temp < tmp
	for j in {0..2}; do
		t=${temp[${j}]}
		((scores[j]+=t))
		((variance[j]+=t*t))
	done
done
rm tmp
echo "${scores[@]}"
echo "${variance[@]}"
python sim_results.py "${scores[@]}" "${variance[@]}" "$n"

