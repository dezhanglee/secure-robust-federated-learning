#!/bin/bash

COUNTER=0


for DATASET in "MNIST"
do
	for ATTACK in "backdoor" "krum" "modelpoisoning" "noattack" "trimmedmean"
	do
		for AGG in "average"  "ex_noregret" "filterl2" "median" "randomized"
		do
			DEVICE=$((COUNTER%3))
			COMMAND="python src/simulate_random_agg.py --dataset ${DATASET} --device ${DEVICE} --attack ${ATTACK} --agg ${AGG} > log_eval/${ATTACK}_${AGG}_${DATASET}.log;"
            echo ${COMMAND}
            bash -c "conda init; ${COMMAND}"
			COUNTER=$((COUNTER+1))
		done
	done
done
