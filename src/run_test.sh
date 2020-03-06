#!/bin/bash
for num_epoch in $(seq 1 5)
do
	./train_model  --output_file_serialized=final_serialized_5_2000.txt --input_file=final_serialized_5_2000.txt --epochs_max=1000 --topology=2 6 6 1 >> process_5_2000.log
	./analyze_model --output_file_serialized=final_serialized_5_2000.txt --input_file=final_serialized_5_2000.txt
done
