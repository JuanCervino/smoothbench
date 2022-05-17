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

#python -m smooth.scripts.navigation_manifold  --algorithm ERM --n_train 25 --epochs 100000 --n_unlab 100 --hidden_neurons 128\
#                                              --lr 0.0001 --weight_decay 0 --dataset Dijkstra_grid_maze --width 0.05\
#                                              --regularizer 1 --heat_kernel_t 0.6
#
python -m smooth.scripts.navigation_manifold  --algorithm ERM --n_train 25 --epochs 300000 --n_unlab 100 --hidden_neurons 128\
                                              --lr 0.0002 --weight_decay 0 --dataset Dijkstra_grid_maze_two_points --width 0.05\
                                              --regularizer 0.1 --heat_kernel_t 0.2 --dual_step_mu 0.1 --dual_step_lambda 0.1 --epsilon 0.2\
                                              --heat_kernel_t 0.1 --clamp 0.4


#python -m smooth.scripts.navigation_manifold  --algorithm LIPSCHITZ_NO_RHO --n_train 25 --epochs 300000 --n_unlab 100 --hidden_neurons 128\
#                                              --lr 0.0002 --weight_decay 0 --dataset Dijkstra_grid_maze_two_points --width 0.05\
#                                              --regularizer 0.1 --heat_kernel_t 0.2 --dual_step_mu 0.1 --dual_step_lambda 0.1 --epsilon 0.2\
#                                              --heat_kernel_t 0.1 --clamp 0.3