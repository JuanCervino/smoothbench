# search_dir=~/Documents/NumericalResults/advbench/train-output-baselines
# for entry in "$search_dir"/*
# do
#   echo $(basename $entry)
#   echo $(basename $(dirname $entry))
#   python -m advbench.scripts.collect_results --depth 0 --input_dir train-output-baselines/$(basename $entry)
#   python -m advbench.plotting.learning_curve --input_dir train-output-baselines/$(basename $entry)
#   python -m advbench.plotting.acc_and_loss --input_dir train-output-baselines/$(basename $entry)

# done

search_dir=~/Documents/Github/smoothbench/train-output
for entry in "$search_dir"/*
do
  echo $(basename $entry)
  echo $(basename $(dirname $entry))
  python -m smooth.scripts.collect_results --depth 0 --input_dir train-output/$(basename $entry)
  python -m smooth.plotting.learning_curve --input_dir train-output/$(basename $entry)
  python -m smooth.plotting.acc_and_loss --input_dir train-output/$(basename $entry)

done
