#!/bin/bash
for num_epoch in $(seq 1 5)
do
	./train_model --epochs_max=2000 >> process.log
	./analyze_model
	cp updated_model.txt final_result_serialized.txt
done
