#python -m smooth.scripts.navigation_manifold  --algorithm ERM --n_train 7 --epochs 10000 --n_unlab 100 --hidden_neurons 128\
#                                              --lr 0.0001 --weight_decay 0 --dataset Dijkstra_grid_window --width 0.25



#python -m smooth.scripts.navigation_manifold  --algorithm ERM --n_train 24 --epochs 30000 --n_unlab 100 --hidden_neurons 128\
#                                              --lr 0.0001 --weight_decay 0 --dataset Dijkstra_grid_window --width 0.1
#
#
#python -m smooth.scripts.navigation_manifold  --algorithm ERM --n_train 31 --epochs 30000 --n_unlab 100 --hidden_neurons 128\
#                                              --lr 0.0001 --weight_decay 0 --dataset Dijkstra_grid_window --width 0.1\
#                                              --regularizer 0.05
#
#python -m smooth.scripts.navigation_manifold  --algorithm LAPLACIAN_REGULARIZATION --n_train 31 --epochs 30000 --n_unlab 100 --hidden_neurons 128\
#                                              --lr 0.0001 --weight_decay 0 --dataset Dijkstra_grid_window --width 0.1\
#                                              --regularizer 0.05

python -m smooth.scripts.navigation_manifold  --algorithm ERM --n_train 31 --epochs 100000 --n_unlab 100 --hidden_neurons 128\
                                              --lr 0.0001 --weight_decay 0 --dataset Dijkstra_grid_maze --width 0.1\
                                              --regularizer 1 --heat_kernel_t 0.6

#python -m smooth.scripts.navigation_manifold  --algorithm LAPLACIAN_REGULARIZATION --n_train 31 --epochs 200000 --n_unlab 100 --hidden_neurons 128\
#                                              --lr 0.0001 --weight_decay 0 --dataset Dijkstra_grid_maze --width 0.1\
#                                              --regularizer 3.8 --heat_kernel_t 0.6
#
#python -m smooth.scripts.navigation_manifold  --algorithm LAPLACIAN_REGULARIZATION --n_train 31 --epochs 200000 --n_unlab 100 --hidden_neurons 128\
#                                              --lr 0.0001 --weight_decay 0 --dataset Dijkstra_grid_maze --width 0.1\
#                                              --regularizer 3.9 --heat_kernel_t 0.6
#
#python -m smooth.scripts.navigation_manifold  --algorithm LAPLACIAN_REGULARIZATION --n_train 31 --epochs 200000 --n_unlab 100 --hidden_neurons 128\
#                                              --lr 0.0001 --weight_decay 0 --dataset Dijkstra_grid_maze --width 0.1\
#                                              --regularizer 3.8 --heat_kernel_t 0.7
#
#python -m smooth.scripts.navigation_manifold  --algorithm LAPLACIAN_REGULARIZATION --n_train 31 --epochs 200000 --n_unlab 100 --hidden_neurons 128\
#                                              --lr 0.0001 --weight_decay 0 --dataset Dijkstra_grid_maze --width 0.1\
#                                              --regularizer 3.9 --heat_kernel_t 0.5



#python -m smooth.scripts.navigation_manifold  --algorithm LAPLACIAN_REGULARIZATION --n_train 31 --epochs 30000 --n_unlab 100 --hidden_neurons 128\
#                                              --lr 0.0001 --weight_decay 0 --dataset Dijkstra_grid_maze --width 0.1\
#                                              --regularizer 1 --heat_kernel_t 0.8
#python -m smooth.scripts.navigation_manifold  --algorithm LAPLACIAN_REGULARIZATION --n_train 31 --epochs 30000 --n_unlab 100 --hidden_neurons 128\
#                                              --lr 0.0001 --weight_decay 0 --dataset Dijkstra_grid_maze --width 0.1\
#                                              --regularizer 2 --heat_kernel_t 0.8
#python -m smooth.scripts.navigation_manifold  --algorithm LAPLACIAN_REGULARIZATION --n_train 31 --epochs 30000 --n_unlab 100 --hidden_neurons 128\
#                                              --lr 0.0001 --weight_decay 0 --dataset Dijkstra_grid_maze --width 0.1\
#                                              --regularizer 3 --heat_kernel_t 0.8
#python -m smooth.scripts.navigation_manifold  --algorithm LAPLACIAN_REGULARIZATION --n_train 31 --epochs 30000 --n_unlab 100 --hidden_neurons 128\
#                                              --lr 0.0001 --weight_decay 0 --dataset Dijkstra_grid_maze --width 0.1\
#                                              --regularizer 4 --heat_kernel_t 0.8
#python -m smooth.scripts.navigation_manifold  --algorithm LAPLACIAN_REGULARIZATION --n_train 31 --epochs 30000 --n_unlab 100 --hidden_neurons 128\
#                                              --lr 0.0001 --weight_decay 0 --dataset Dijkstra_grid_maze --width 0.1\
#                                              --regularizer 5 --heat_kernel_t 0.8
#python -m smooth.scripts.navigation_manifold  --algorithm LAPLACIAN_REGULARIZATION --n_train 31 --epochs 30000 --n_unlab 100 --hidden_neurons 128\
#                                              --lr 0.0001 --weight_decay 0 --dataset Dijkstra_grid_maze --width 0.1\
#                                              --regularizer 6 --heat_kernel_t 0.8
#python -m smooth.scripts.navigation_manifold  --algorithm LAPLACIAN_REGULARIZATION --n_train 31 --epochs 30000 --n_unlab 100 --hidden_neurons 128\
#                                              --lr 0.0001 --weight_decay 0 --dataset Dijkstra_grid_maze --width 0.1\
#                                              --regularizer 7 --heat_kernel_t 0.8

# ---------------------------------------------------------------
#
#python -m smooth.scripts.navigation_manifold  --algorithm LAPLACIAN_REGULARIZATION --n_train 31 --epochs 30000 --n_unlab 100 --hidden_neurons 128\
#                                              --lr 0.0001 --weight_decay 0 --dataset Dijkstra_grid_maze --width 0.1\
#                                              --regularizer 2 --heat_kernel_t 0.8
#python -m smooth.scripts.navigation_manifold  --algorithm LAPLACIAN_REGULARIZATION --n_train 31 --epochs 30000 --n_unlab 100 --hidden_neurons 128\
#                                              --lr 0.0001 --weight_decay 0 --dataset Dijkstra_grid_maze --width 0.1\
#                                              --regularizer 2 --heat_kernel_t 1
#python -m smooth.scripts.navigation_manifold  --algorithm LAPLACIAN_REGULARIZATION --n_train 31 --epochs 30000 --n_unlab 100 --hidden_neurons 128\
#                                              --lr 0.0001 --weight_decay 0 --dataset Dijkstra_grid_maze --width 0.1\
#                                              --regularizer 2 --heat_kernel_t 1.8
#python -m smooth.scripts.navigation_manifold  --algorithm LAPLACIAN_REGULARIZATION --n_train 31 --epochs 30000 --n_unlab 100 --hidden_neurons 128\
#                                              --lr 0.0001 --weight_decay 0 --dataset Dijkstra_grid_maze --width 0.1\
#                                              --regularizer 2 --heat_kernel_t 2.8
#python -m smooth.scripts.navigation_manifold  --algorithm LAPLACIAN_REGULARIZATION --n_train 31 --epochs 30000 --n_unlab 100 --hidden_neurons 128\
#                                              --lr 0.0001 --weight_decay 0 --dataset Dijkstra_grid_maze --width 0.1\
#                                              --regularizer 2 --heat_kernel_t 3.8
#python -m smooth.scripts.navigation_manifold  --algorithm LAPLACIAN_REGULARIZATION --n_train 31 --epochs 30000 --n_unlab 100 --hidden_neurons 128\
#                                              --lr 0.0001 --weight_decay 0 --dataset Dijkstra_grid_maze --width 0.1\
#                                              --regularizer 2 --heat_kernel_t 5.8
#python -m smooth.scripts.navigation_manifold  --algorithm LAPLACIAN_REGULARIZATION --n_train 31 --epochs 30000 --n_unlab 100 --hidden_neurons 128\
#                                              --lr 0.0001 --weight_decay 0 --dataset Dijkstra_grid_maze --width 0.1\
#                                              --regularizer 2 --heat_kernel_t 10.5
#python -m smooth.scripts.navigation_manifold  --algorithm LAPLACIAN_REGULARIZATION --n_train 31 --epochs 30000 --n_unlab 100 --hidden_neurons 128\
#                                              --lr 0.0001 --weight_decay 0 --dataset Dijkstra_grid_maze --width 0.1\
#                                              --regularizer 2 --heat_kernel_t 15.0

# ---------------------------------------------------------------
#
#python -m smooth.scripts.navigation_manifold  --algorithm LAPLACIAN_REGULARIZATION --n_train 31 --epochs 30000 --n_unlab 100 --hidden_neurons 128\
#                                              --lr 0.0001 --weight_decay 0 --dataset Dijkstra_grid_maze --width 0.1\
#                                              --regularizer 3 --heat_kernel_t 0.8
#python -m smooth.scripts.navigation_manifold  --algorithm LAPLACIAN_REGULARIZATION --n_train 31 --epochs 30000 --n_unlab 100 --hidden_neurons 128\
#                                              --lr 0.0001 --weight_decay 0 --dataset Dijkstra_grid_maze --width 0.1\
#                                              --regularizer 3 --heat_kernel_t 1
#python -m smooth.scripts.navigation_manifold  --algorithm LAPLACIAN_REGULARIZATION --n_train 31 --epochs 30000 --n_unlab 100 --hidden_neurons 128\
#                                              --lr 0.0001 --weight_decay 0 --dataset Dijkstra_grid_maze --width 0.1\
#                                              --regularizer 3 --heat_kernel_t 1.8
#python -m smooth.scripts.navigation_manifold  --algorithm LAPLACIAN_REGULARIZATION --n_train 31 --epochs 30000 --n_unlab 100 --hidden_neurons 128\
#                                              --lr 0.0001 --weight_decay 0 --dataset Dijkstra_grid_maze --width 0.1\
#                                              --regularizer 3 --heat_kernel_t 2.8
#python -m smooth.scripts.navigation_manifold  --algorithm LAPLACIAN_REGULARIZATION --n_train 31 --epochs 30000 --n_unlab 100 --hidden_neurons 128\
#                                              --lr 0.0001 --weight_decay 0 --dataset Dijkstra_grid_maze --width 0.1\
#                                              --regularizer 3 --heat_kernel_t 3.8
#python -m smooth.scripts.navigation_manifold  --algorithm LAPLACIAN_REGULARIZATION --n_train 31 --epochs 30000 --n_unlab 100 --hidden_neurons 128\
#                                              --lr 0.0001 --weight_decay 0 --dataset Dijkstra_grid_maze --width 0.1\
#                                              --regularizer 3 --heat_kernel_t 5.8
#python -m smooth.scripts.navigation_manifold  --algorithm LAPLACIAN_REGULARIZATION --n_train 31 --epochs 30000 --n_unlab 100 --hidden_neurons 128\
#                                              --lr 0.0001 --weight_decay 0 --dataset Dijkstra_grid_maze --width 0.1\
#                                              --regularizer 3 --heat_kernel_t 10.5
#python -m smooth.scripts.navigation_manifold  --algorithm LAPLACIAN_REGULARIZATION --n_train 31 --epochs 30000 --n_unlab 100 --hidden_neurons 128\
#                                              --lr 0.0001 --weight_decay 0 --dataset Dijkstra_grid_maze --width 0.1\
#                                              --regularizer 3 --heat_kernel_t 15.0
