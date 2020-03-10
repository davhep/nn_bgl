#!/bin/bash
outdir=datarez
mkdir $outdir
numseries=$1
sizeseries=$2
filepostfix=${numseries}_${sizeseries}
serialezed=$outdir/final_serialized_$filepostfix.txt
errfile=$outdir/err_$filepostfix.log
#rm $outdir/final_serialized_$filepostfix.txt
#rm $outdir/process_$filepostfix.log
#rm $errfile
for num_epoch in $(seq 1 $numseries)
do
	./train_model  --output_file_serialized=$outdir/final_serialized_$filepostfix.txt --input_file=$serialezed --output_file_dot=$outdir/final_${filepostfix}_epoch_${num_epoch}.dot --epochs_max=$sizeseries --topology=2 6 6 1 >> $outdir/process_$filepostfix.log 2>>$errfile
	./analyze_model --output_file_serialized=$outdir/final_serialized_$filepostfix.txt --input_file=$serialezed >>$errfile 2>>$errfile
done
