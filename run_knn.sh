# python -m smooth.scripts.compute_knn --dataset CIFAR10 
python -m smooth.scripts.compute_knn --dataset CIFAR10  --transforms normalized --metric euclidean --pretrained imagenet --model alexnet

# python -m smooth.scripts.compute_knn --dataset CIFAR10  --transforms None --metric euclidean

# python -m smooth.scripts.compute_knn --dataset CIFAR10  --transforms Normalized --metric cosine_similarity
# python -m smooth.scripts.compute_knn --dataset CIFAR10  --transforms None --metric cosine_similarity