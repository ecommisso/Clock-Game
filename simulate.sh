#!/bin/bash

#iterations
n=$1
#players
p1=$2
p2=$3
p3=$4

#don't touch these
s1=()
s2=()
s3=()

for ((i=0 ; i < n ; i++)); do
	printf "$p1\n$p2\n$p3\n" | bash clock_game.sh True $RANDOM &&
		sleep 1
	sed -n '26,28 p' log_moves.txt | awk '{print $5}' >> tmp
done
readarray -t temp < tmp
rm tmp
echo "${temp[@]}"
python sim_results.py ${temp[@]} $n

