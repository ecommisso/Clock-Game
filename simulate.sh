#!/bin/bash
#iterations
n=$1
#players
p1=$2
p2=$3
p3=$4

for i in `seq 0 $n`
do
	printf "$p1\n$p2\n$p3\n" | bash clock_game.sh True $RANDOM &&
		sleep 1
	sed -n '26,28 p' log_moves.txt | awk '{print $5}' >> tmp
done
readarray -t temp < tmp
rm tmp
echo "${temp[@]}"
python sim_results.py ${temp[@]} $n

