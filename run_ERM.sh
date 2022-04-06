# python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM --output_dir train-output --normalize True --per_labeled 0.1
# python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM --output_dir train-output --normalize True --per_labeled 0.2
# python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM --output_dir train-output --normalize True --per_labeled 0.3
# python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM --output_dir train-output --normalize True --per_labeled 0.4
# python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM --output_dir train-output --normalize True --per_labeled 0.5
# python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM --output_dir train-output --normalize True --per_labeled 0.6
# python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM --output_dir train-output --normalize True --per_labeled 0.7
# python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM --output_dir train-output --normalize True --per_labeled 0.8
# python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM --output_dir train-output --normalize True --per_labeled 0.9
# python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM --output_dir train-output --normalize True --per_labeled 1

# 
#python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP --output_dir train-output --per_labeled 0.1  --normalize False --regularizer 0.1 --heat_kernel_t 10.55
# python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP_KNN --output_dir train-output --per_labeled 0.1  --normalize True --regularizer 0.1 --heat_kernel_t 10.55
# python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP_KNN --output_dir train-output --per_labeled 0.1  --normalize True --regularizer 0.1 --heat_kernel_t 10.55
# python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP_KNN --output_dir train-output --per_labeled 0.1  --normalize True --regularizer 0.1 --heat_kernel_t 10.55
# python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP_KNN --output_dir train-output --per_labeled 0.1  --normalize True --regularizer 0.1 --heat_kernel_t 10.55
# python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP_KNN --output_dir train-output --per_labeled 0.1  --normalize True --regularizer 0.1 --heat_kernel_t 10.55
# python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP_KNN --output_dir train-output --per_labeled 0.1  --normalize True --regularizer 0.1 --heat_kernel_t 10.55
# python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP_KNN --output_dir train-output --per_labeled 0.1  --normalize True --regularizer 0.1 --heat_kernel_t 10.55


# python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP --output_dir train-output --per_labeled 0.1  --normalize False --regularizer 0.2 --heat_kernel_t 10.55
# python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP --output_dir train-output --per_labeled 0.1  --normalize True --regularizer 0.2 --heat_kernel_t 10.55

# python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP --output_dir train-output --per_labeled 0.1  --normalize False --regularizer 0.3 --heat_kernel_t 10.55
# python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP --output_dir train-output --per_labeled 0.1  --normalize True --regularizer 0.3 --heat_kernel_t 10.55

# python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP --output_dir train-output --per_labeled 0.1  --normalize False --regularizer 0.25 --heat_kernel_t 10.55
# python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP --output_dir train-output --per_labeled 0.1  --normalize True --regularizer 0.25 --heat_kernel_t 10.55

# python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP --output_dir train-output --normalize True --regularizer 1. --per_labeled 0.3 
# python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP_KNN --output_dir train-output --normalize True --regularizer .01 --per_labeled 0.001 --heat_kernel_t 10.55 --unlabeled_batch_size 1
# python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP_KNN --output_dir train-output --normalize True --regularizer 0.1 --per_labeled 0.001 --heat_kernel_t 10.55 --unlabeled_batch_size 10 
# python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP_KNN --output_dir train-output 
# python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP_CNN_METRIC --output_dir train-output 
python -m smooth.scripts.train_no_attack\
			--dataset CIFAR10 --algorithm ERM_AVG_LIP_KNN --output_dir train-output\
			 --precalculated_folder knn-baselines2/resnet184_None_cosine_similarity_2022-0401-190338\
			  --k 3 --normalize True --regularizer 0.1 --heat_kernel_t 10.5 --unlab_augmentation 5\
			  --per_labeled 0.1 --unlab_batch_size 10


# python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP_CNN_METRIC --output_dir train-output 
# python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP_CNN_METRIC --output_dir train-output 
# python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP_CNN_METRIC --output_dir train-output 


#python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP_TRANSFORM --output_dir train-output --normalize True --regularizer 0.005 --per_labeled 0.001 --heat_kernel_t 10 --unlabeled_batch_size 100
#
#python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP_TRANSFORM --output_dir train-output --normalize False --regularizer 0.005 --per_labeled 0.001 --heat_kernel_t 10 --unlabeled_batch_size 200
#python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP_TRANSFORM --output_dir train-output --normalize True --regularizer 0.005 --per_labeled 0.001 --heat_kernel_t 10 --unlabeled_batch_size 200
#
#python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP_TRANSFORM --output_dir train-output --normalize False --regularizer 0.005 --per_labeled 0.001 --heat_kernel_t 5 --unlabeled_batch_size 100
#python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP_TRANSFORM --output_dir train-output --normalize True --regularizer 0.005 --per_labeled 0.001 --heat_kernel_t 5 --unlabeled_batch_size 100
#
#python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP_TRANSFORM --output_dir train-output --normalize False --regularizer 0.005 --per_labeled 0.001 --heat_kernel_t 5 --unlabeled_batch_size 200
#python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP_TRANSFORM --output_dir train-output --normalize True --regularizer 0.005 --per_labeled 0.001 --heat_kernel_t 5 --unlabeled_batch_size 200
#
#python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP_TRANSFORM --output_dir train-output --normalize False --regularizer 0.01 --per_labeled 0.001 --heat_kernel_t 10 --unlabeled_batch_size 100
#python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP_TRANSFORM --output_dir train-output --normalize True --regularizer 0.01 --per_labeled 0.001 --heat_kernel_t 10 --unlabeled_batch_size 100
#
#python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP_TRANSFORM --output_dir train-output --normalize False --regularizer 0.01 --per_labeled 0.001 --heat_kernel_t 10 --unlabeled_batch_size 200
#python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP_TRANSFORM --output_dir train-output --normalize True --regularizer 0.01 --per_labeled 0.001 --heat_kernel_t 10 --unlabeled_batch_size 200
#
#python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP_TRANSFORM --output_dir train-output --normalize False --regularizer 0.01 --per_labeled 0.001 --heat_kernel_t 5 --unlabeled_batch_size 100
#python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP_TRANSFORM --output_dir train-output --normalize True --regularizer 0.01 --per_labeled 0.001 --heat_kernel_t 5 --unlabeled_batch_size 100
#
#python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP_TRANSFORM --output_dir train-output --normalize False --regularizer 0.01 --per_labeled 0.001 --heat_kernel_t 5 --unlabeled_batch_size 200
#python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP_TRANSFORM --output_dir train-output --normalize True --regularizer 0.01 --per_labeled 0.001 --heat_kernel_t 5 --unlabeled_batch_size 200
#
#python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP_TRANSFORM --output_dir train-output --normalize False --regularizer 0.001 --per_labeled 0.001 --heat_kernel_t 10 --unlabeled_batch_size 100
#python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP_TRANSFORM --output_dir train-output --normalize True --regularizer 0.001 --per_labeled 0.001 --heat_kernel_t 10 --unlabeled_batch_size 100
#
#python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP_TRANSFORM --output_dir train-output --normalize False --regularizer 0.001 --per_labeled 0.001 --heat_kernel_t 10 --unlabeled_batch_size 200
#python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP_TRANSFORM --output_dir train-output --normalize True --regularizer 0.001 --per_labeled 0.001 --heat_kernel_t 10 --unlabeled_batch_size 200
#
#python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP_TRANSFORM --output_dir train-output --normalize False --regularizer 0.001 --per_labeled 0.001 --heat_kernel_t 5 --unlabeled_batch_size 100
#python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP_TRANSFORM --output_dir train-output --normalize True --regularizer 0.001 --per_labeled 0.001 --heat_kernel_t 5 --unlabeled_batch_size 100
#
#python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP_TRANSFORM --output_dir train-output --normalize False --regularizer 0.001 --per_labeled 0.001 --heat_kernel_t 5 --unlabeled_batch_size 200
#python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP_TRANSFORM --output_dir train-output --normalize True --regularizer 0.001 --per_labeled 0.001 --heat_kernel_t 5 --unlabeled_batch_size 200

# python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP_TRANSFORM --output_dir train-output --normalize False --regularizer 0.01 --per_labeled 0.001 --heat_kernel_t 20 --unlabeled_batch_size 100
# python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP_TRANSFORM --output_dir train-output --normalize False --regularizer 0.01 --per_labeled 0.001 --heat_kernel_t 30 --unlabeled_batch_size 100
# python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP_TRANSFORM --output_dir train-output --normalize False --regularizer 0.01 --per_labeled 0.001 --heat_kernel_t 15 --unlabeled_batch_size 100




# python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP_CHEAT --output_dir train-output --normalize False --regularizer 0.2 --per_labeled 0.001 --heat_kernel_t 10 --unlabeled_batch_size 1
# python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP_CHEAT --output_dir train-output --normalize False --regularizer 0.2 --per_labeled 0.001 --heat_kernel_t 15 --unlabeled_batch_size 1
# python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP_CHEAT --output_dir train-output --normalize False --regularizer 0.2 --per_labeled 0.001 --heat_kernel_t 20 --unlabeled_batch_size 1
# python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP_CHEAT --output_dir train-output --normalize False --regularizer 0.2 --per_labeled 0.001 --heat_kernel_t 25 --unlabeled_batch_size 1

# python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP_CHEAT --output_dir train-output --normalize False --regularizer 0.1 --per_labeled 0.001 --heat_kernel_t 5 --unlabeled_batch_size 1
# python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP_CHEAT --output_dir train-output --normalize False --regularizer 0.1 --per_labeled 0.001 --heat_kernel_t 10 --unlabeled_batch_size 1
# python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP_CHEAT --output_dir train-output --normalize False --regularizer 0.1 --per_labeled 0.001 --heat_kernel_t 15 --unlabeled_batch_size 1
# python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP_CHEAT --output_dir train-output --normalize False --regularizer 0.1 --per_labeled 0.001 --heat_kernel_t 20 --unlabeled_batch_size 1
# python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP_CHEAT --output_dir train-output --normalize False --regularizer 0.1 --per_labeled 0.001 --heat_kernel_t 25 --unlabeled_batch_size 1

# python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP_CHEAT --output_dir train-output --normalize False --regularizer 0.3 --per_labeled 0.001 --heat_kernel_t 5 --unlabeled_batch_size 1
# python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP_CHEAT --output_dir train-output --normalize False --regularizer 0.3 --per_labeled 0.001 --heat_kernel_t 10 --unlabeled_batch_size 1
# python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP_CHEAT --output_dir train-output --normalize False --regularizer 0.3 --per_labeled 0.001 --heat_kernel_t 15 --unlabeled_batch_size 1
# python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP_CHEAT --output_dir train-output --normalize False --regularizer 0.3 --per_labeled 0.001 --heat_kernel_t 20 --unlabeled_batch_size 1
# python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP_CHEAT --output_dir train-output --normalize False --regularizer 0.3 --per_labeled 0.001 --heat_kernel_t 25 --unlabeled_batch_size 1

# python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP_CHEAT --output_dir train-output --normalize False --regularizer 0.4 --per_labeled 0.001 --heat_kernel_t 5 --unlabeled_batch_size 1
# python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP_CHEAT --output_dir train-output --normalize False --regularizer 0.4 --per_labeled 0.001 --heat_kernel_t 10 --unlabeled_batch_size 1
# python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP_CHEAT --output_dir train-output --normalize False --regularizer 0.4 --per_labeled 0.001 --heat_kernel_t 15 --unlabeled_batch_size 1
# python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP_CHEAT --output_dir train-output --normalize False --regularizer 0.4 --per_labeled 0.001 --heat_kernel_t 20 --unlabeled_batch_size 1
# python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP_CHEAT --output_dir train-output --normalize False --regularizer 0.4 --per_labeled 0.001 --heat_kernel_t 25 --unlabeled_batch_size 1

# python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP_CHEAT --output_dir train-output --normalize False --regularizer 0. --per_labeled 0.001 --heat_kernel_t 5 --unlabeled_batch_size 1
# python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP_CHEAT --output_dir train-output --normalize False --regularizer 0. --per_labeled 0.001 --heat_kernel_t 10 --unlabeled_batch_size 1
# python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP_CHEAT --output_dir train-output --normalize False --regularizer 0. --per_labeled 0.001 --heat_kernel_t 15 --unlabeled_batch_size 1
# python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP_CHEAT --output_dir train-output --normalize False --regularizer 0. --per_labeled 0.001 --heat_kernel_t 20 --unlabeled_batch_size 1
# python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP_CHEAT --output_dir train-output --normalize False --regularizer 0. --per_labeled 0.001 --heat_kernel_t 25 --unlabeled_batch_size 1

# python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP_CHEAT --output_dir train-output --normalize False --regularizer 0.2 --per_labeled 0.001 --heat_kernel_t 5 --unlabeled_batch_size 1
# python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP_CHEAT --output_dir train-output --normalize False --regularizer 0.2 --per_labeled 0.001 --heat_kernel_t 10 --unlabeled_batch_size 1
# python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP_CHEAT --output_dir train-output --normalize False --regularizer 0.2 --per_labeled 0.001 --heat_kernel_t 15 --unlabeled_batch_size 1
# python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP_CHEAT --output_dir train-output --normalize False --regularizer 0.2 --per_labeled 0.001 --heat_kernel_t 20 --unlabeled_batch_size 1
# python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP_CHEAT --output_dir train-output --normalize False --regularizer 0.2 --per_labeled 0.001 --heat_kernel_t 25 --unlabeled_batch_size 1

# python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP_CHEAT --output_dir train-output --normalize False --regularizer 0.1 --per_labeled 0.001 --heat_kernel_t 5 --unlabeled_batch_size 1
# python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP_CHEAT --output_dir train-output --normalize False --regularizer 0.1 --per_labeled 0.001 --heat_kernel_t 10 --unlabeled_batch_size 1
# python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP_CHEAT --output_dir train-output --normalize False --regularizer 0.1 --per_labeled 0.001 --heat_kernel_t 15 --unlabeled_batch_size 1
# python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP_CHEAT --output_dir train-output --normalize False --regularizer 0.1 --per_labeled 0.001 --heat_kernel_t 20 --unlabeled_batch_size 1
# python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP_CHEAT --output_dir train-output --normalize False --regularizer 0.1 --per_labeled 0.001 --heat_kernel_t 25 --unlabeled_batch_size 1

# python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP_CHEAT --output_dir train-output --normalize False --regularizer 0.3 --per_labeled 0.001 --heat_kernel_t 5 --unlabeled_batch_size 1
# python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP_CHEAT --output_dir train-output --normalize False --regularizer 0.3 --per_labeled 0.001 --heat_kernel_t 10 --unlabeled_batch_size 1
# python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP_CHEAT --output_dir train-output --normalize False --regularizer 0.3 --per_labeled 0.001 --heat_kernel_t 15 --unlabeled_batch_size 1
# python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP_CHEAT --output_dir train-output --normalize False --regularizer 0.3 --per_labeled 0.001 --heat_kernel_t 20 --unlabeled_batch_size 1
# python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP_CHEAT --output_dir train-output --normalize False --regularizer 0.3 --per_labeled 0.001 --heat_kernel_t 25 --unlabeled_batch_size 1

# python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP_CHEAT --output_dir train-output --normalize False --regularizer 0.4 --per_labeled 0.001 --heat_kernel_t 5 --unlabeled_batch_size 1
# python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP_CHEAT --output_dir train-output --normalize False --regularizer 0.4 --per_labeled 0.001 --heat_kernel_t 10 --unlabeled_batch_size 1
# python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP_CHEAT --output_dir train-output --normalize False --regularizer 0.4 --per_labeled 0.001 --heat_kernel_t 15 --unlabeled_batch_size 1
# python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP_CHEAT --output_dir train-output --normalize False --regularizer 0.4 --per_labeled 0.001 --heat_kernel_t 20 --unlabeled_batch_size 1
# python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP_CHEAT --output_dir train-output --normalize False --regularizer 0.4 --per_labeled 0.001 --heat_kernel_t 25 --unlabeled_batch_size 1

# python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP_CHEAT --output_dir train-output --normalize False --regularizer 0. --per_labeled 0.001 --heat_kernel_t 5 --unlabeled_batch_size 1
# python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP_CHEAT --output_dir train-output --normalize False --regularizer 0. --per_labeled 0.001 --heat_kernel_t 10 --unlabeled_batch_size 1
# python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP_CHEAT --output_dir train-output --normalize False --regularizer 0. --per_labeled 0.001 --heat_kernel_t 15 --unlabeled_batch_size 1
# python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP_CHEAT --output_dir train-output --normalize False --regularizer 0. --per_labeled 0.001 --heat_kernel_t 20 --unlabeled_batch_size 1
# python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP_CHEAT --output_dir train-output --normalize False --regularizer 0. --per_labeled 0.001 --heat_kernel_t 25 --unlabeled_batch_size 1

#python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP_KNN --output_dir train-output --normalize True --regularizer 0.01 --per_labeled 0.001 --heat_kernel_t 30 --unlabeled_batch_size 1



# python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP_KNN --output_dir train-output --normalize True --regularizer .01 --per_labeled 0.001 --heat_kernel_t 10.55 --unlabeled_batch_size 64

# python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP_KNN --output_dir train-output --normalize True --regularizer 1. --per_labeled 0.001 --heat_kernel_t 10.55 --unlabeled_batch_size 32
# python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP_KNN --output_dir train-output --normalize True --regularizer 0.5 --per_labeled 0.001 --heat_kernel_t 10.55 --unlabeled_batch_size 32
# python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP_KNN --output_dir train-output --normalize True --regularizer 3. --per_labeled 0.001 --heat_kernel_t 10.55 --unlabeled_batch_size 32
# python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP_KNN --output_dir train-output --normalize True --regularizer 10. --per_labeled 0.001 --heat_kernel_t 10.55 --unlabeled_batch_size 32

# python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP --output_dir train-output --normalize True --regularizer 5. --per_labeled 0.1 
# python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP --output_dir train-output --normalize True --regularizer 5. --per_labeled 0.3 
# python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP --output_dir train-output --normalize True --regularizer 5. --per_labeled 0.5 

# python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP --output_dir train-output --normalize True --regularizer 10. --per_labeled 0.1 
# python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP --output_dir train-output --normalize True --regularizer 10. --per_labeled 0.3 
# python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP --output_dir train-output --normalize True --regularizer 10. --per_labeled 0.5 

# python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP --output_dir train-output --normalize True --regularizer 100. --per_labeled 0.1
# python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP --output_dir train-output --normalize True --regularizer 100. --per_labeled 0.3 
# python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP --output_dir train-output --normalize True --regularizer 100. --per_labeled 0.5

# python -m smooth.scripts.search_t --dataset CIFAR10 --algorithm ERM_AVG_LIP --output_dir train-output --normalize True --regularizer 100. --per_labeled 0.5

# python -m smooth.scripts.train_no_attack --dataset MNIST --algorithm ERM_LAMBDA_LIP --output_dir train-output --normalize True --regularizer 10. --per_labeled 0.1 --unlabeled_batch_size 256

