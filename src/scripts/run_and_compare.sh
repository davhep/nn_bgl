#!/bin/bash
for num_epoch in $(seq 1 1000)
do
	./train_model --epochs_max=1000 >> process.log
	#cp best_result.dot final_result_$num_epoch.dot
	#cp best_result_serialized.txt final_result_serialized.txt
	./analyze_model
	cp updated_model.txt final_result_serialized.txt
done
