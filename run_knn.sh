# python -m smooth.scripts.compute_knn --dataset CIFAR10 
python -m smooth.scripts.compute_knn --dataset CIFAR10  --transforms Normalized --metric Euclidean
python -m smooth.scripts.compute_knn --dataset CIFAR10  --transforms None --metric Euclidean

python -m smooth.scripts.compute_knn --dataset CIFAR10  --transforms Normalized --metric cosine_similarity
python -m smooth.scripts.compute_knn --dataset CIFAR10  --transforms None --metric cosine_similarity