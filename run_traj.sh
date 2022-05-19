
##
python -m smooth.scripts.navigation_manifold  --algorithm ERM --n_train 25 --epochs 300000 --n_unlab 100 --hidden_neurons 128\
                                              --lr 0.0002 --weight_decay 0 --dataset Dijkstra_grid_maze_two_points --width 0.05\
                                              --regularizer 0.1 --heat_kernel_t 0.2 --dual_step_mu 0.1 --dual_step_lambda 0.1 --epsilon 0.2\
                                              --heat_kernel_t 0.1 --clamp 0.3 --radius 0.5
#
#
python -m smooth.scripts.navigation_manifold  --algorithm ERM --n_train 25 --epochs 300000 --n_unlab 100 --hidden_neurons 128\
                                              --lr 0.0002 --weight_decay 0.1 --dataset Dijkstra_grid_maze_two_points --width 0.05\
                                              --regularizer 0.1 --heat_kernel_t 0.2 --dual_step_mu 0.1 --dual_step_lambda 0.1 --epsilon 0.2\
                                              --heat_kernel_t 0.1 --clamp 0.3 --radius 0.5

python -m smooth.scripts.navigation_manifold  --algorithm LAPLACIAN_REGULARIZATION --n_train 25 --epochs 300000 --n_unlab 100 --hidden_neurons 128\
                                              --lr 0.0002 --weight_decay 0 --dataset Dijkstra_grid_maze_two_points --width 0.05\
                                              --regularizer 0.0001 --heat_kernel_t 0.2 --dual_step_mu 0.1 --dual_step_lambda 0.1 --epsilon 0.2\
                                              --heat_kernel_t 0.1 --clamp 0.3 --radius 0.5




#
python -m smooth.scripts.navigation_manifold  --algorithm LAPLACIAN_REGULARIZATION --n_train 25 --epochs 300000 --n_unlab 100 --hidden_neurons 128\
                                              --lr 0.0002 --weight_decay 0 --dataset Dijkstra_grid_maze_two_points --width 0.05\
                                              --regularizer 0.01 --heat_kernel_t 0.2 --dual_step_mu 0.1 --dual_step_lambda 0.1 --epsilon 0.2\
                                              --heat_kernel_t 0.1 --clamp 0.3 --radius 0.5

python -m smooth.scripts.navigation_manifold  --algorithm LIPSCHITZ_NO_RHO --n_train 25 --epochs 300000 --n_unlab 100 --hidden_neurons 128\
                                              --lr 0.0002 --weight_decay 0 --dataset Dijkstra_grid_maze_two_points --width 0.05\
                                              --regularizer 0.1 --heat_kernel_t 0.2 --dual_step_mu 0.1 --dual_step_lambda 0.1 --epsilon 0.1\
                                              --heat_kernel_t 0.1 --clamp 0.3 --radius 0.5

python -m smooth.scripts.navigation_manifold  --algorithm LIPSCHITZ_NO_RHO --n_train 25 --epochs 300000 --n_unlab 100 --hidden_neurons 128\
                                              --lr 0.0002 --weight_decay 0 --dataset Dijkstra_grid_maze_two_points --width 0.05\
                                              --regularizer 0.1 --heat_kernel_t 0.2 --dual_step_mu 0.1 --dual_step_lambda 0.1 --epsilon 0.1\
                                              --heat_kernel_t 0.1 --clamp 0.3 --radius 0.5

python -m smooth.scripts.navigation_manifold  --algorithm LIPSCHITZ_NO_RHO --n_train 25 --epochs 300000 --n_unlab 100 --hidden_neurons 128\
                                              --lr 0.0002 --weight_decay 0 --dataset Dijkstra_grid_maze_two_points --width 0.05\
                                              --regularizer 0.1 --heat_kernel_t 0.2 --dual_step_mu 0.1 --dual_step_lambda 0.1 --epsilon 0.1\
                                              --heat_kernel_t 0.1 --clamp 0.3 --radius 0.5
#
