# search_dir=~/Documents/NumericalResults/advbench/train-output-baselines
# for entry in "$search_dir"/*
# do
#   echo $(basename $entry)
#   echo $(basename $(dirname $entry))
#   python -m advbench.scripts.collect_results --depth 0 --input_dir train-output-baselines/$(basename $entry)
#   python -m advbench.plotting.learning_curve --input_dir train-output-baselines/$(basename $entry)
#   python -m advbench.plotting.acc_and_loss --input_dir train-output-baselines/$(basename $entry)

# done
# python  -m smooth.scripts.aggrupate_results --depth 0 --input_dir train-output-baselines2/ERM_0.1_2022-0310-155121 --file_to_write concat

search_dir=~/Documents/NumericalResults/smooth/train-output-baselines2
for entry in "$search_dir"/*
do
  echo $(basename $entry)
  echo $(basename $(dirname $entry))
  python -m smooth.scripts.aggrupate_results --depth 0 --input_dir train-output-baselines2/$(basename $entry) --file_to_write concat
  # python -m smooth.plotting.learning_curve --input_dir train-output-baselines2/$(basename $entry)
  # python -m smooth.plotting.acc_and_loss --input_dir train-output-baselines2/$(basename $entry)

done
