# python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM --output_dir train-output --normalize True --per_labeled 0.1
# python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM --output_dir train-output --normalize True --per_labeled 0.2
# python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM --output_dir train-output --normalize True --per_labeled 0.3
# python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM --output_dir train-output --normalize True --per_labeled 0.4
# python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM --output_dir train-output --normalize True --per_labeled 0.5
# python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM --output_dir train-output --normalize True --per_labeled 0.6
# python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM --output_dir train-output --normalize True --per_labeled 0.7
# python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM --output_dir train-output --normalize True --per_labeled 0.8
# python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM --output_dir train-output --normalize True --per_labeled 0.9
python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM --output_dir train-output --normalize True --per_labeled 1

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
# python -m smooth.scripts.train_no_attack --dataset CIFAR10 --algorithm ERM_AVG_LIP --output_dir train-output --normalize True --regularizer 1. --per_labeled 0.5 

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

