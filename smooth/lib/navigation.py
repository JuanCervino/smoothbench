import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cp
import pickle,os
import scipy
from matplotlib.patches import Rectangle
import torch

# x_center,y_i,y_f
def sample_out_maze(sample, width, tolerance):
    out = True
    if sample[0] < 5+width+tolerance and sample[0] > 5-width-tolerance and sample[1] > 3 - tolerance:
        out = False
    if sample[0] < 15+width+tolerance and sample[0] > 15-width-tolerance and sample[1] < 7 +tolerance:
        out = False
    return out

def get_poly_vector(t,degree):
    vec = np.ones(degree+1)
    aux = t
    vec[1] = t
    for d in range(2,degree+1):
        aux = aux * t
        vec[d] = aux

    return vec

def step(state,acc,time_step):
    """
    Step
    state = x, y, x_dot, y_dot
    acc = x_dotdot, y_dotdot
    """
    new_state = np.zeros(len(state))
    if len(state) == 4:
        new_state[0] = state[0] + state[2]*time_step + acc[0]*time_step**2 * 0.5
        new_state[1] = state[1] + state[3]*time_step + acc[1]*time_step**2 * 0.5
        new_state[2] = state[2] + acc[0]*time_step
        new_state[3] = state[3] + acc[1]*time_step
    elif len(state) == 2:
        new_state[0] = state[0] + acc[0]*time_step
        new_state[1] = state[1] + acc[1]*time_step
    return new_state


def generate_trajectory(start, goal, intermediate_points, degree_poly, total_time, points):

    first = np.linspace(0, degree_poly, degree_poly + 1)
    second = np.multiply(first[1:-1], first[2::])
    second = np.concatenate((np.zeros(2), second))
    args_col = np.array([[np.linspace(1, degree_poly - 1, degree_poly - 1)] * (degree_poly - 1)])[0, :, :]
    args_row = np.array([[np.linspace(0, degree_poly - 2, degree_poly - 1)] * (degree_poly - 1)])[0, :, :].T
    div = np.ones((degree_poly + 1, degree_poly + 1))
    div[2::, 2::] = args_row + args_col
    vec_time = get_poly_vector(total_time, degree_poly)
    p_vector = np.multiply(second, vec_time)
    P = np.outer(p_vector, p_vector)
    P = np.divide(P, div)
    P = P + 0.01 * np.eye(degree_poly + 1)

    # Initial and final position with no velocity, and no acceleration
    eval_0 = np.concatenate((np.ones(1), np.zeros(degree_poly)))
    Ai = np.concatenate((eval_0, np.roll(eval_0, 1), np.roll(eval_0, 2)))
    Ai = Ai.reshape(3, degree_poly + 1)
    bix = np.array([start[0], 0, 0])
    biy = np.array([start[1], 0, 0])

    vec = get_poly_vector(total_time, degree_poly)
    Af = np.concatenate((vec, np.multiply(first, vec), np.multiply(vec, second)))
    Af = Af.reshape(3, degree_poly + 1)
    bfx = np.array([goal[0], 0, 0])
    bfy = np.array([goal[1], 0, 0])

    # Intermediate Points
    if intermediate_points != None:
        Ax = np.array([])
        Ay = np.array([])

        bx = np.array(intermediate_points)[:, 0]
        by = np.array(intermediate_points)[:, 1]
        for i, p in enumerate(intermediate_points):
            time = (i + 1) / (len(intermediate_points) + 2) * total_time
            vec = get_poly_vector(time, degree_poly)
            Ax = np.concatenate((Ax, vec))
            Ay = np.concatenate((Ay, vec))
        Ax = Ax.reshape(int(len(Ay) / (degree_poly + 1)), degree_poly + 1)
        Ay = Ax.reshape(int(len(Ay) / (degree_poly + 1)), degree_poly + 1)
        x = cp.Variable(degree_poly + 1)
        y = cp.Variable(degree_poly + 1)
        prob = cp.Problem(cp.Minimize((1 / 2) * cp.quad_form(x, P) + (1 / 2) * cp.quad_form(y, P)),
                          [Ax @ x <= bx + 0.5,
                           Ax @ x >= bx - 0.5,
                           Ay @ y <= by + 0.5,
                           Ay @ y >= by - 0.5,
                           Ai @ x == bix,
                           Af @ x == bfx,
                           Ai @ y == biy,
                           Af @ y == bfy])
                          # [Ax @ x == bx,
                          #  Ay @ y == by,
                          #  Ai @ x == bix,
                          #  Af @ x == bfx,
                          #  Ai @ y == biy,
                          #  Af @ y == bfy])
        prob.solve()
    else:
        x = cp.Variable(degree_poly + 1)
        y = cp.Variable(degree_poly + 1)
        prob = cp.Problem(cp.Minimize((1 / 2) * cp.quad_form(x, P) + (1 / 2) * cp.quad_form(y, P)),
                           [Ai @ x == bix,
                           Af @ x == bfx,
                           Ai @ y == biy,
                           Af @ y == bfy])
        prob.solve()


    labeled_times = np.linspace(0, total_time, points + 2)
    labeled_times = labeled_times[1:-1]

    labeled_pos_poly = [get_poly_vector(t, degree_poly) for t in labeled_times]
    labeled_vel_poly = [np.multiply(first, np.roll(labeled_pos_poly_vec, 1)) for labeled_pos_poly_vec in
                        labeled_pos_poly]
    labeled_acc_poly = [np.multiply(second, np.roll(labeled_pos_poly_vec, 2)) for labeled_pos_poly_vec in
                        labeled_pos_poly]
    # print(x.value)
    labeled_points_x = np.array([x.value @ poly for poly in labeled_pos_poly])
    labeled_points_y = np.array([y.value @ poly for poly in labeled_pos_poly])
    labeled_vel_x = np.array([x.value @ poly for poly in labeled_vel_poly])
    labeled_vel_y = np.array([y.value @ poly for poly in labeled_vel_poly])
    labeled_acc_x = np.array([x.value @ poly for poly in labeled_acc_poly])
    labeled_acc_y = np.array([y.value @ poly for poly in labeled_acc_poly])

    X_lab = np.column_stack((labeled_points_x, labeled_points_y, labeled_vel_x, labeled_vel_y))
    y_lab = np.column_stack((labeled_acc_x, labeled_acc_y))
    return X_lab,y_lab

# def generate_dijkstra():

def create_dataset(dataset, n_dim, n_train, n_unlab,  data_dir, width, resolution=0.4, n_test=100):

    assert dataset in ['window','center','Dijkstra_grid_window','Dijkstra_grid_maze','Dijkstra_grid_maze_two_points','Dijkstra_grid_window']
    file_name = '/'+dataset+'_train'+str(n_train)+'_unlab'+str(n_unlab)+'_width'+str(width)+'.p'

    total_time = 4
    # Train Set
    if os.path.exists(data_dir+file_name):
        # [X_lab,y_lab,X_unlab, y_unlab] = pickle.load(data_dir+file_name)
        with open(data_dir+file_name, 'rb') as pickle_file:
            [X_lab,y_lab,X_unlab, y_unlab,adj_matrix] = pickle.load(pickle_file)
    else:
        if dataset == 'window':
            goal = [19,1]
            degree_poly = 8


            start = [1,1]
            # intermediate_points = [[5,3],[10,5],[15,3]]
            intermediate_points = [[10,5]]
            intermediate_points = None

            [X_lab, y_lab] = generate_trajectory(start,goal,intermediate_points,degree_poly,total_time,n_train)

            start = [18,7]
            intermediate_points = [[18.5,3.5]]
            intermediate_points = None
            [X_unlab, y_unlab] = generate_trajectory(start,goal,intermediate_points,degree_poly,total_time,n_train)


            start = [1,7]
            intermediate_points = [[10,5]]
            intermediate_points = None
            [X_unlab_aux, y_unlab_aux] = generate_trajectory(start,goal,intermediate_points,degree_poly,total_time,n_train)
            X_unlab = np.vstack((X_unlab,X_unlab_aux))
            y_unlab = np.vstack((y_unlab,y_unlab_aux))


            start = [3,1]
            intermediate_points = [[10,5]]
            intermediate_points = None
            [X_unlab_aux, y_unlab_aux] = generate_trajectory(start,goal,intermediate_points,degree_poly,total_time,n_train)
            X_unlab = np.vstack((X_unlab,X_unlab_aux))
            y_unlab = np.vstack((y_unlab,y_unlab_aux))

            start = [2,3]
            intermediate_points = [[10,5]]
            intermediate_points = None
            [X_unlab_aux, y_unlab_aux] = generate_trajectory(start,goal,intermediate_points,degree_poly,total_time,n_train)
            X_unlab = np.vstack((X_unlab,X_unlab_aux))
            y_unlab = np.vstack((y_unlab,y_unlab_aux))


            start = [7,9]
            intermediate_points = [[7,5],[13,5]]
            intermediate_points = None

            [X_unlab_aux, y_unlab_aux] = generate_trajectory(start,goal,intermediate_points,degree_poly,total_time,n_train)
            X_unlab = np.vstack((X_unlab,X_unlab_aux))
            y_unlab = np.vstack((y_unlab,y_unlab_aux))

            start = [10,5]
            intermediate_points = [[13,5]]
            intermediate_points = None
            [X_unlab_aux, y_unlab_aux] = generate_trajectory(start,goal,intermediate_points,degree_poly,total_time,n_train)
            X_unlab = np.vstack((X_unlab,X_unlab_aux))
            y_unlab = np.vstack((y_unlab,y_unlab_aux))

            start = [11,5]
            intermediate_points = [[13,5]]
            intermediate_points = None
            [X_unlab_aux, y_unlab_aux] = generate_trajectory(start,goal,intermediate_points,degree_poly,total_time,n_train)
            X_unlab = np.vstack((X_unlab,X_unlab_aux))
            y_unlab = np.vstack((y_unlab,y_unlab_aux))

            start = [11,6]
            intermediate_points = [[13,5]]
            intermediate_points = None
            [X_unlab_aux, y_unlab_aux] = generate_trajectory(start,goal,intermediate_points,degree_poly,total_time,n_train)
            X_unlab = np.vstack((X_unlab,X_unlab_aux))
            y_unlab = np.vstack((y_unlab,y_unlab_aux))

            start = [15,3]
            intermediate_points = [[13,5]]
            intermediate_points = None
            [X_unlab_aux, y_unlab_aux] = generate_trajectory(start,goal,intermediate_points,degree_poly,total_time,n_train)
            X_unlab = np.vstack((X_unlab,X_unlab_aux))
            y_unlab = np.vstack((y_unlab,y_unlab_aux))

            start = [12,1]
            intermediate_points = [[13,5]]
            intermediate_points = None
            [X_unlab_aux, y_unlab_aux] = generate_trajectory(start,goal,intermediate_points,degree_poly,total_time,n_train)
            X_unlab = np.vstack((X_unlab,X_unlab_aux))
            y_unlab = np.vstack((y_unlab,y_unlab_aux))

            start = [13,4]
            intermediate_points = [[13,5]]
            intermediate_points = None
            [X_unlab_aux, y_unlab_aux] = generate_trajectory(start,goal,intermediate_points,degree_poly,total_time,n_train)
            X_unlab = np.vstack((X_unlab,X_unlab_aux))
            y_unlab = np.vstack((y_unlab,y_unlab_aux))

            start = [13,1]
            intermediate_points = [[13,5]]
            intermediate_points = None
            [X_unlab_aux, y_unlab_aux] = generate_trajectory(start,goal,intermediate_points,degree_poly,total_time,n_train)
            X_unlab = np.vstack((X_unlab,X_unlab_aux))
            y_unlab = np.vstack((y_unlab,y_unlab_aux))

            start = [11,1.5]
            intermediate_points = [[13,5]]
            intermediate_points = None
            [X_unlab_aux, y_unlab_aux] = generate_trajectory(start,goal,intermediate_points,degree_poly,total_time,n_train)
            X_unlab = np.vstack((X_unlab,X_unlab_aux))
            y_unlab = np.vstack((y_unlab,y_unlab_aux))

            start = [10,7]
            intermediate_points = [[13,5]]
            intermediate_points = None
            [X_unlab_aux, y_unlab_aux] = generate_trajectory(start,goal,intermediate_points,degree_poly,total_time,n_train)
            X_unlab = np.vstack((X_unlab,X_unlab_aux))
            y_unlab = np.vstack((y_unlab,y_unlab_aux))

            start = [2.5,5]
            # intermediate_points = [[10,5]]
            intermediate_points = None
            [X_unlab_aux, y_unlab_aux] = generate_trajectory(start,goal,intermediate_points,degree_poly,total_time,n_train)
            X_unlab = np.vstack((X_unlab,X_unlab_aux))
            y_unlab = np.vstack((y_unlab,y_unlab_aux))

            start = [12,3]
            intermediate_points = None
            [X_unlab_aux, y_unlab_aux] = generate_trajectory(start,goal,intermediate_points,degree_poly,total_time,n_train)
            X_unlab = np.vstack((X_unlab,X_unlab_aux))
            y_unlab = np.vstack((y_unlab,y_unlab_aux))


            start = [15,8]
            intermediate_points = None
            [X_unlab_aux, y_unlab_aux] = generate_trajectory(start,goal,intermediate_points,degree_poly,total_time,n_train)
            X_unlab = np.vstack((X_unlab,X_unlab_aux))
            y_unlab = np.vstack((y_unlab,y_unlab_aux))

            start = [15,4]
            intermediate_points = None
            # intermediate_points = [[10,5]]
            [X_unlab_aux, y_unlab_aux] = generate_trajectory(start,goal,intermediate_points,degree_poly,total_time,n_train)
            X_unlab = np.vstack((X_unlab,X_unlab_aux))
            y_unlab = np.vstack((y_unlab,y_unlab_aux))

            start = [12.5,8]
            intermediate_points = None
            [X_unlab_aux, y_unlab_aux] = generate_trajectory(start,goal,intermediate_points,degree_poly,total_time,n_train)
            X_unlab = np.vstack((X_unlab,X_unlab_aux))
            y_unlab = np.vstack((y_unlab,y_unlab_aux))

            start = [7,3]
            intermediate_points = None
            # intermediate_points = [[10,5]] # Correct
            [X_unlab_aux, y_unlab_aux] = generate_trajectory(start,goal,intermediate_points,degree_poly,total_time,n_train)
            X_unlab = np.vstack((X_unlab,X_unlab_aux))
            y_unlab = np.vstack((y_unlab,y_unlab_aux))



            X_lab = np.vstack((X_lab,X_unlab))
            y_lab = np.vstack((y_lab,y_unlab))

        elif dataset == 'center':
            goal = [10, 5]
            degree_poly = 8

            start = [1, 1]
            # intermediate_points = [[5,3],[10,5],[15,3]]
            intermediate_points = [[10, 5]]
            intermediate_points = None

            [X_lab, y_lab] = generate_trajectory(start, goal, intermediate_points, degree_poly, total_time, n_train)

            start = [18, 7]
            intermediate_points = [[18.5, 3.5]]
            intermediate_points = None
            [X_unlab, y_unlab] = generate_trajectory(start, goal, intermediate_points, degree_poly, total_time,
                                                     n_train)

            start = [1, 7]
            intermediate_points = [[10, 5]]
            intermediate_points = None
            [X_unlab_aux, y_unlab_aux] = generate_trajectory(start, goal, intermediate_points, degree_poly,
                                                             total_time, n_train)
            X_unlab = np.vstack((X_unlab, X_unlab_aux))
            y_unlab = np.vstack((y_unlab, y_unlab_aux))

            start = [3, 1]
            intermediate_points = [[10, 5]]
            intermediate_points = None
            [X_unlab_aux, y_unlab_aux] = generate_trajectory(start, goal, intermediate_points, degree_poly,
                                                             total_time, n_train)
            X_unlab = np.vstack((X_unlab, X_unlab_aux))
            y_unlab = np.vstack((y_unlab, y_unlab_aux))

            start = [2, 3]
            intermediate_points = [[10, 5]]
            intermediate_points = None
            [X_unlab_aux, y_unlab_aux] = generate_trajectory(start, goal, intermediate_points, degree_poly,
                                                             total_time, n_train)
            X_unlab = np.vstack((X_unlab, X_unlab_aux))
            y_unlab = np.vstack((y_unlab, y_unlab_aux))

            start = [7, 9]
            intermediate_points = [[7, 5], [13, 5]]
            intermediate_points = None

            [X_unlab_aux, y_unlab_aux] = generate_trajectory(start, goal, intermediate_points, degree_poly,
                                                             total_time, n_train)
            X_unlab = np.vstack((X_unlab, X_unlab_aux))
            y_unlab = np.vstack((y_unlab, y_unlab_aux))

            start = [10, 5]
            intermediate_points = [[13, 5]]
            intermediate_points = None
            [X_unlab_aux, y_unlab_aux] = generate_trajectory(start, goal, intermediate_points, degree_poly,
                                                             total_time, n_train)
            X_unlab = np.vstack((X_unlab, X_unlab_aux))
            y_unlab = np.vstack((y_unlab, y_unlab_aux))

            start = [11, 5]
            intermediate_points = [[13, 5]]
            intermediate_points = None
            [X_unlab_aux, y_unlab_aux] = generate_trajectory(start, goal, intermediate_points, degree_poly,
                                                             total_time, n_train)
            X_unlab = np.vstack((X_unlab, X_unlab_aux))
            y_unlab = np.vstack((y_unlab, y_unlab_aux))

            start = [11, 6]
            intermediate_points = [[13, 5]]
            intermediate_points = None
            [X_unlab_aux, y_unlab_aux] = generate_trajectory(start, goal, intermediate_points, degree_poly,
                                                             total_time, n_train)
            X_unlab = np.vstack((X_unlab, X_unlab_aux))
            y_unlab = np.vstack((y_unlab, y_unlab_aux))

            start = [15, 3]
            intermediate_points = [[13, 5]]
            intermediate_points = None
            [X_unlab_aux, y_unlab_aux] = generate_trajectory(start, goal, intermediate_points, degree_poly,
                                                             total_time, n_train)
            X_unlab = np.vstack((X_unlab, X_unlab_aux))
            y_unlab = np.vstack((y_unlab, y_unlab_aux))

            start = [12, 1]
            intermediate_points = [[13, 5]]
            intermediate_points = None
            [X_unlab_aux, y_unlab_aux] = generate_trajectory(start, goal, intermediate_points, degree_poly,
                                                             total_time, n_train)
            X_unlab = np.vstack((X_unlab, X_unlab_aux))
            y_unlab = np.vstack((y_unlab, y_unlab_aux))

            start = [13, 4]
            intermediate_points = [[13, 5]]
            intermediate_points = None
            [X_unlab_aux, y_unlab_aux] = generate_trajectory(start, goal, intermediate_points, degree_poly,
                                                             total_time, n_train)
            X_unlab = np.vstack((X_unlab, X_unlab_aux))
            y_unlab = np.vstack((y_unlab, y_unlab_aux))

            start = [13, 1]
            intermediate_points = [[13, 5]]
            intermediate_points = None
            [X_unlab_aux, y_unlab_aux] = generate_trajectory(start, goal, intermediate_points, degree_poly,
                                                             total_time, n_train)
            X_unlab = np.vstack((X_unlab, X_unlab_aux))
            y_unlab = np.vstack((y_unlab, y_unlab_aux))

            start = [11, 1.5]
            intermediate_points = [[13, 5]]
            intermediate_points = None
            [X_unlab_aux, y_unlab_aux] = generate_trajectory(start, goal, intermediate_points, degree_poly,
                                                             total_time, n_train)
            X_unlab = np.vstack((X_unlab, X_unlab_aux))
            y_unlab = np.vstack((y_unlab, y_unlab_aux))

            start = [10, 7]
            intermediate_points = [[13, 5]]
            intermediate_points = None
            [X_unlab_aux, y_unlab_aux] = generate_trajectory(start, goal, intermediate_points, degree_poly,
                                                             total_time, n_train)
            X_unlab = np.vstack((X_unlab, X_unlab_aux))
            y_unlab = np.vstack((y_unlab, y_unlab_aux))

            start = [2.5, 5]
            # intermediate_points = [[10,5]]
            intermediate_points = None
            [X_unlab_aux, y_unlab_aux] = generate_trajectory(start, goal, intermediate_points, degree_poly,
                                                             total_time, n_train)
            X_unlab = np.vstack((X_unlab, X_unlab_aux))
            y_unlab = np.vstack((y_unlab, y_unlab_aux))

            start = [12, 3]
            intermediate_points = None
            [X_unlab_aux, y_unlab_aux] = generate_trajectory(start, goal, intermediate_points, degree_poly,
                                                             total_time, n_train)
            X_unlab = np.vstack((X_unlab, X_unlab_aux))
            y_unlab = np.vstack((y_unlab, y_unlab_aux))

            start = [15, 8]
            intermediate_points = None
            [X_unlab_aux, y_unlab_aux] = generate_trajectory(start, goal, intermediate_points, degree_poly,
                                                             total_time, n_train)
            X_unlab = np.vstack((X_unlab, X_unlab_aux))
            y_unlab = np.vstack((y_unlab, y_unlab_aux))

            start = [15, 4]
            intermediate_points = None
            # intermediate_points = [[10,5]]
            [X_unlab_aux, y_unlab_aux] = generate_trajectory(start, goal, intermediate_points, degree_poly,
                                                             total_time, n_train)
            X_unlab = np.vstack((X_unlab, X_unlab_aux))
            y_unlab = np.vstack((y_unlab, y_unlab_aux))

            start = [12.5, 8]
            intermediate_points = None
            [X_unlab_aux, y_unlab_aux] = generate_trajectory(start, goal, intermediate_points, degree_poly,
                                                             total_time, n_train)
            X_unlab = np.vstack((X_unlab, X_unlab_aux))
            y_unlab = np.vstack((y_unlab, y_unlab_aux))

            start = [7, 3]
            intermediate_points = None
            # intermediate_points = [[10,5]] # Correct
            [X_unlab_aux, y_unlab_aux] = generate_trajectory(start, goal, intermediate_points, degree_poly,
                                                             total_time, n_train)
            X_unlab = np.vstack((X_unlab, X_unlab_aux))
            y_unlab = np.vstack((y_unlab, y_unlab_aux))

            X_lab = np.vstack((X_lab, X_unlab))
            y_lab = np.vstack((y_lab, y_unlab))

        elif dataset == 'Dijkstra_grid_window':

            # Construct graph
            x = np.linspace(0, 20, 2*n_train-1)
            y = np.linspace(0, 10, n_train)
            # full coorindate arrays
            xx, yy = np.meshgrid(x, y)
            graph = np.column_stack((xx.flatten(),yy.flatten()))

            # Add Start And Stop
            start = np.array([8,1])
            stop = np.array([19,1])
            graph = np.vstack((start,graph,stop))

            list_of_delete = []
            # print(graph.shape)

            for sample in range(graph.shape[0]):
                if not ((graph[sample, 0] < 10 - (width + resolution*0.5) or graph[sample, 0] > 10 + (
                        width + resolution)) or (
                                graph[sample, 1] > (4 + resolution) and graph[sample, 1] < (6 - resolution))):
                    list_of_delete = list_of_delete + [sample]


            graph = np.delete(graph,list_of_delete,axis=0)
            matrix_graph = scipy.spatial.distance.cdist(graph,graph)
            matrix_graph = np.where(matrix_graph > 0.01+np.sqrt((10/(n_train-1))**2+(10/(n_train-1))**2), 0, matrix_graph)
            # matrix_graph = np.where(matrix_graph > 2.5, 0, matrix_graph)

            matrix_graph_sparse = scipy.sparse.csr_matrix(matrix_graph)
            _, predecessors = scipy.sparse.csgraph.dijkstra(csgraph=matrix_graph_sparse, directed=False, indices=0, return_predecessors=True)
            adj_matrix, _ = scipy.sparse.csgraph.dijkstra(csgraph=matrix_graph_sparse, directed=False, return_predecessors=True)

            cx = scipy.sparse.coo_matrix(matrix_graph_sparse)
            fig, ax = plt.subplots()
            ax.add_patch(Rectangle((10 - width, 0), 2 * width, 4,
                                  edgecolor='black',
                                  facecolor='black',
                                  fill=True,
                                  lw=5))

            ax.add_patch(Rectangle((10 - width, 6), 2 * width, 4,
                                  edgecolor='black',
                                  facecolor='black',
                                  fill=True,
                                  lw=5))
            for i, j, v in zip(cx.row, cx.col, cx.data):
               arr = np.vstack((graph[i,:],graph[j,:]))
               plt.plot(arr[:,0],arr[:,1],'b-')
            plt.plot(graph[:,0],graph[:,1],'*')

            path = [len(predecessors)-1]
            item = predecessors[-1]
            path = path + [item]
            while item != 0:
                item = predecessors[item]
                path = path + [item]

            plt.plot(graph[path,0],graph[path,1],'r')

            plt.plot(graph[0,0],graph[0,1],'r*')
            plt.plot(graph[-1,0],graph[-1,1],'ro')

            # plt.show()
            X_lab = np.array(graph[path[1:],:])
            y_lab = np.array(graph[path[:-1]]-graph[path[1:]])/0.1
            X_unlab = graph
            y_unlab = None

        elif dataset == 'Dijkstra_random_window':
           # Construct graph
           graph = np.array([19,1])
           weights = np.array([20,10]) # Max x, max y, max x_dot, max y_dot
           for sample in range(n_unlab):
               s = np.multiply(np.random.uniform(0, 1, 2),weights)
               while not ((s[0] < 10-width or s[0] > 10 + width) or (s[1]>4 and s[1]<6)):
                   s = np.multiply(np.random.uniform(0, 1, 2), weights)

               graph = np.vstack((s,graph))
           graph = np.vstack((np.array([1,1]), graph))
           matrix_graph = scipy.spatial.distance.cdist(graph,graph)
           matrix_graph = np.where(matrix_graph > -0.01+(1/n_train)*np.sqrt((10)**2+(20)**2), 0, matrix_graph)

           matrix_graph_sparse = scipy.sparse.csr_matrix(matrix_graph)
           dist_matrix, predecessors = scipy.sparse.csgraph.dijkstra(csgraph=matrix_graph_sparse, directed=False, indices=0, return_predecessors=True)
           cx = scipy.sparse.coo_matrix(matrix_graph_sparse)
           fig, ax = plt.subplots()
           ax.add_patch(Rectangle((10 - width, 0), 2 * width, 4,
                                  edgecolor='black',
                                  facecolor='black',
                                  fill=True,
                                  lw=5))

           ax.add_patch(Rectangle((10 - width, 6), 2 * width, 4,
                                  edgecolor='black',
                                  facecolor='black',
                                  fill=True,
                                  lw=5))
           for i, j, v in zip(cx.row, cx.col, cx.data):
               arr = np.vstack((graph[i,:],graph[j,:]))
               plt.plot(arr[:,0],arr[:,1],'g-')
           plt.plot(graph[:,0],graph[:,1],'*')

           plt.plot(graph[0,0],graph[0,1],'r*')
           plt.plot(graph[-1,0],graph[-1,1],'ro')


           plt.show()

        elif dataset == 'Dijkstra_grid_maze':

            # Construct graph
            x = np.linspace(0, 20, 2 * n_train - 1)
            y = np.linspace(0, 10, n_train)
            # full coorindate arrays
            xx, yy = np.meshgrid(x, y)
            graph_unlab = np.column_stack((xx.flatten(), yy.flatten()))
            list_of_delete = []
            # print(graph.shape)

            for sample in range(graph_unlab.shape[0]):
                if not sample_out_maze(graph_unlab[sample, :], width, resolution):
                    list_of_delete = list_of_delete + [sample]

            graph_unlab = np.delete(graph_unlab, list_of_delete, axis=0)


            x = np.linspace(2.5, 17.5, 5)
            y = np.linspace(1, 10, 8)
            xx, yy = np.meshgrid(x, y)

            start_points = np.column_stack((xx.flatten(), yy.flatten()))
            stop = np.array([19, 1])
            X_lab = []
            for count, start in enumerate(start_points):
                # Add Start And Stop
            # print(start)
                start = np.array(start)
                graph = np.vstack((start, graph_unlab, stop))
                matrix_graph = scipy.spatial.distance.cdist(graph, graph)

                matrix_graph = np.where(
                    matrix_graph > 0.01 + np.sqrt((10 / (n_train - 1)) ** 2 + (10 / (n_train - 1)) ** 2), 0,
                    matrix_graph)
                # matrix_graph = np.where(matrix_graph > 2.5, 0, matrix_graph)

                matrix_graph = scipy.sparse.csr_matrix(matrix_graph)
                # list_of_delete = []
                # # print(graph.shape)
                #
                # for sample in range(graph.shape[0]):
                #    if not sample_out_maze(graph[sample,:],width,resolution):
                #        list_of_delete = list_of_delete + [sample]
                #
                #
                # graph = np.delete(graph, list_of_delete, axis=0)
                # matrix_graph = scipy.spatial.distance.cdist(graph_unlab, graph_unlab)
                # matrix_graph= np.where(
                #     matrix_graph> 0.01 + np.sqrt((10 / (n_train - 1)) ** 2 + (10 / (n_train - 1)) ** 2), 0, matrix_graph)
                # # matrix_graph = np.where(matrix_graph > 2.5, 0, matrix_graph)
                #
                # matrix_graph= scipy.sparse.csr_matrix(matrix_graph)
                _, predecessors = scipy.sparse.csgraph.dijkstra(csgraph=matrix_graph, directed=False, indices=0,
                                                                return_predecessors=True)
                adj_matrix, _ = scipy.sparse.csgraph.dijkstra(csgraph=matrix_graph, directed=False,
                                                              return_predecessors=True)

                cx = scipy.sparse.coo_matrix(matrix_graph)
                if count == 0:
                    fig, ax = plt.subplots()
                    ax.add_patch(Rectangle((5-width, 3), 2*width, 7,
                                           edgecolor='black',
                                           facecolor='black',
                                           fill=True,
                                           lw=5))

                    ax.add_patch(Rectangle((15-width, 0), 2*width, 7,
                                           edgecolor='black',
                                           facecolor='black',
                                           fill=True,
                                           lw=5))
                    for i, j, v in zip(cx.row, cx.col, cx.data):
                        arr = np.vstack((graph[i, :], graph[j, :]))
                        plt.plot(arr[:, 0], arr[:, 1], 'b-')
                    plt.plot(graph[:, 0], graph[:, 1], '*')

                path = [len(predecessors) - 1]
                item = predecessors[-1]
                path = path + [item]
                while item != 0:
                    item = predecessors[item]
                    path = path + [item]

                plt.plot(graph[path, 0], graph[path, 1], 'r')

                plt.plot(graph[0, 0], graph[0, 1], 'r*')
                plt.plot(graph[-1, 0], graph[-1, 1], 'ro')

            # plt.show()
                if X_lab == []:
                    X_lab = np.array(graph[path[1:], :])
                    y_lab = np.array(graph[path[:-1]] - graph[path[1:]]) / 0.1
                else:
                    X_lab = np.vstack((X_lab,np.array(graph[path[1:], :])))
                    y_lab = np.vstack((y_lab,np.array(graph[path[:-1]] - graph[path[1:]]) / 0.1 ))
            X_unlab = graph_unlab
            y_unlab = None

        elif dataset == 'Dijkstra_grid_maze_two_points':

            # Construct graph
            x = np.linspace(0, 20, 2 * n_train - 1)
            y = np.linspace(0, 10, n_train)
            # full coorindate arrays
            xx, yy = np.meshgrid(x, y)
            graph_unlab = np.column_stack((xx.flatten(), yy.flatten()))
            list_of_delete = []
            # print(graph.shape)

            for sample in range(graph_unlab.shape[0]):
                if not sample_out_maze(graph_unlab[sample, :], width, resolution):
                    list_of_delete = list_of_delete + [sample]

            graph_unlab = np.delete(graph_unlab, list_of_delete, axis=0)


            # x = np.linspace(2.5, 17.5, 5)
            # y = np.linspace(1, 10, 8)
            # xx, yy = np.meshgrid(x, y)
            #
            # start_points = np.column_stack((xx.flatten(), yy.flatten()))
            start_points = [[1,9],[14,1]]
            stop = np.array([19, 1])
            X_lab = []
            for count, start in enumerate(start_points):
                # Add Start And Stop
            # print(start)
                start = np.array(start)
                graph = np.vstack((start, graph_unlab, stop))
                matrix_graph = scipy.spatial.distance.cdist(graph, graph)

                matrix_graph = np.where(
                    matrix_graph > 0.01 + np.sqrt((10 / (n_train - 1)) ** 2 + (10 / (n_train - 1)) ** 2), 0,
                    matrix_graph)
                # matrix_graph = np.where(matrix_graph > 2.5, 0, matrix_graph)

                matrix_graph = scipy.sparse.csr_matrix(matrix_graph)
                # list_of_delete = []
                # # print(graph.shape)
                #
                # for sample in range(graph.shape[0]):
                #    if not sample_out_maze(graph[sample,:],width,resolution):
                #        list_of_delete = list_of_delete + [sample]
                #
                #
                # graph = np.delete(graph, list_of_delete, axis=0)
                # matrix_graph = scipy.spatial.distance.cdist(graph_unlab, graph_unlab)
                # matrix_graph= np.where(
                #     matrix_graph> 0.01 + np.sqrt((10 / (n_train - 1)) ** 2 + (10 / (n_train - 1)) ** 2), 0, matrix_graph)
                # # matrix_graph = np.where(matrix_graph > 2.5, 0, matrix_graph)
                #
                # matrix_graph= scipy.sparse.csr_matrix(matrix_graph)
                _, predecessors = scipy.sparse.csgraph.dijkstra(csgraph=matrix_graph, directed=False, indices=0,
                                                                return_predecessors=True)
                adj_matrix, _ = scipy.sparse.csgraph.dijkstra(csgraph=matrix_graph, directed=False,
                                                              return_predecessors=True)

                cx = scipy.sparse.coo_matrix(matrix_graph)
                if count == 0:
                    fig, ax = plt.subplots()
                    ax.add_patch(Rectangle((5-width, 3), 2*width, 7,
                                           edgecolor='black',
                                           facecolor='black',
                                           fill=True,
                                           lw=5))

                    ax.add_patch(Rectangle((15-width, 0), 2*width, 7,
                                           edgecolor='black',
                                           facecolor='black',
                                           fill=True,
                                           lw=5))
                    for i, j, v in zip(cx.row, cx.col, cx.data):
                        arr = np.vstack((graph[i, :], graph[j, :]))
                        plt.plot(arr[:, 0], arr[:, 1], 'b-')
                    plt.plot(graph[:, 0], graph[:, 1], '*')

                path = [len(predecessors) - 1]
                item = predecessors[-1]
                path = path + [item]
                while item != 0:
                    item = predecessors[item]
                    path = path + [item]

                plt.plot(graph[path, 0], graph[path, 1], 'r')

                plt.plot(graph[0, 0], graph[0, 1], 'r*')
                plt.plot(graph[-1, 0], graph[-1, 1], 'ro')

            # plt.show()
                if X_lab == []:
                    X_lab = np.array(graph[path[1:], :])
                    y_lab = np.array(graph[path[:-1]] - graph[path[1:]]) / 0.1
                else:
                    X_lab = np.vstack((X_lab,np.array(graph[path[1:], :])))
                    y_lab = np.vstack((y_lab,np.array(graph[path[:-1]] - graph[path[1:]]) / 0.1 ))
            X_unlab = graph_unlab
            y_unlab = np.array([])

        # Save Dataset
        pickle.dump([X_lab, y_lab, X_unlab, y_unlab,adj_matrix], open(data_dir+file_name, "wb"))


    # Test Set
    file_name = '/'+dataset+'_test'+str(n_train)+'_unlab'+str(n_unlab)+'_width'+str(width)+'.p'
    if os.path.exists(data_dir + file_name):
        with open(data_dir+file_name, 'rb') as pickle_file:
            [X_test] = pickle.load(pickle_file)
    else:
        weights = np.array([20, 10])
        X_test = np.array([])
        for sample in range(n_test):
            s = np.multiply(np.random.uniform(0, 1, 2), weights)
            while not sample_out_maze(s, width, 0.2):
                s = np.multiply(np.random.uniform(0, 1, 2), weights)
            if len(X_test) == 0:
                X_test = s
            else:
                X_test = np.vstack((s, X_test))

        pickle.dump([X_test], open(data_dir + file_name, "wb"))
        # pass
    return [X_lab, y_lab, X_unlab, y_unlab], adj_matrix, X_test

def eval_trajectories(net, initials, width, goal, radius, time_step, total_time, dataset,device,X_unlab,X_lab,output_dir,name, plot):
    assert dataset in ['Dijkstra_grid_maze_two_points']
    # plot = (name % 10000 == 0 )
    trajs = [np.array([]) for i in range(len(initials))]
    vels = [np.array([]) for i in range(len(initials))]
    if dataset == 'Dijkstra_grid_maze_two_points':
        successful_trials = 0
        fig, ax = plt.subplots()

        for i, init in enumerate(initials):
            state = np.array(init)
            trajs[i] = state
            for t in range(int(total_time / time_step)):
                with torch.no_grad():
                    acc = net(torch.Tensor(state).to(device)).cpu().detach().numpy()

                state_new = step(state, acc, time_step)
                if len(vels[i]) == 0:
                    vels[i] = acc

                trajs[i] = np.vstack((trajs[i], state_new))
                vels[i] = np.vstack((vels[i], acc))

                if (state[0]<=5-width and state_new[0]>=5-width and (state[1]>=3 or state_new[1]>=3)) \
                    or (state_new[0] <= 5 - width and state[0] >= 5 - width and (state[1] >= 3 or state_new[1] >= 3)) \
                    or (state[0] <= 5 + width and state_new[0] >= 5 + width and (state[1] >= 3 or state_new[1] >= 3)) \
                    or (state_new[0] <= 5 + width and state[0] >= 5 + width and (state[1] >= 3 or state_new[1] >= 3)) \
                    or (state[0] <= 15 - width and state_new[0] >= 15 - width and (state[1] <= 7 or state_new[1] <= 7)) \
                    or (state_new[0] <= 15 - width and state[0] >= 15 - width and (state[1] <= 7 or state_new[1] <= 7)) \
                    or (state[0] <= 15 + width and state_new[0] >= 15 + width and (state[1] <= 7 or state_new[1] <= 7)) \
                    or (state_new[0] <= 15 + width and state[0] >= 15 + width and (state[1] <= 7 or state_new[1] <= 7)) \
                    or state[0] >= 20 or state[0]<=0 or state[1]>=10 or state[1]<=0 \
                    or np.linalg.norm(state-goal)<radius:
                    break


                state = state_new

                # state = step(state, acc, time_step)
                # if len(vels[i]) == 0:
                #     vels[i] = acc
                #
                # trajs[i] = np.vstack((trajs[i], state))
                # vels[i] = np.vstack((vels[i], acc))

                # if not sample_out_maze(state,width,tolerance=0) or state[0]>20 or \
                #         state[0]<0 or state[1]>10 or state[1]<0 or\
                #         np.linalg.norm(state-goal)<radius:
                #     break
                # projection = step(state, 0.75* acc, time_step)
                # if not sample_out_maze(projection,width,tolerance=0) or projection[0]>20 or \
                #         projection[0]<0 or projection[1]>10 or projection[1]<0:
                #     break
                # projection = step(state, 0.5* acc, time_step)
                # if not sample_out_maze(projection,width,tolerance=0) or projection[0]>20 or \
                #         projection[0]<0 or projection[1]>10 or projection[1]<0:
                #     break
                # projection = step(state, 0.25* acc, time_step)
                # if not sample_out_maze(projection,width,tolerance=0) or projection[0]>20 or \
                #         projection[0]<0 or projection[1]>10 or projection[1]<0:
                #     break
                # projection = step(state, -0.75 * acc, time_step)
                # if not sample_out_maze(projection,width,tolerance=0) or projection[0]>20 or \
                #         projection[0]<0 or projection[1]>10 or projection[1]<0:
                #     break
                # projection = step(state, -0.5* acc, time_step)
                # if not sample_out_maze(projection,width,tolerance=0) or projection[0]>20 or \
                #         projection[0]<0 or projection[1]>10 or projection[1]<0:
                #     break
                # projection = step(state, -0.25* acc, time_step)
                # if not sample_out_maze(projection,width,tolerance=0) or projection[0]>20 or \
                #         projection[0]<0 or projection[1]>10 or projection[1]<0:
                #     break

            if np.linalg.norm(state-goal)<radius:
                successful_trials = successful_trials + 1

            # fig, ax = plt.subplots()
            if plot:
                # print(trajs.shape)
                plt.plot(trajs[i][:, 0], trajs[i][:, 1], '.-')

                ax.plot(goal[0], goal[1], 'r*')
                ax.plot(initials[i][0], initials[i][1], 'g*')
                ax.quiver(X_unlab[:, 0].cpu(), X_unlab[:, 1].cpu(), net(X_unlab).cpu().detach().numpy()[:, 0],
                          net(X_unlab).cpu().detach().numpy()[:, 1],
                          color="#ff0000")  # Blue Unlab
                ax.quiver(X_lab[:, 0].cpu(), X_lab[:, 1].cpu(), net(X_lab).cpu().detach().numpy()[:, 0],
                          net(X_lab).cpu().detach().numpy()[:, 1],
                          color="#0000ff")  # Blue Lab
                ax.plot(initials[i][0], initials[i][1], 'g*')
        if plot:
            ax.add_patch(Rectangle((5 - width, 3), 2 * width, 7,
                                   edgecolor='black',
                                   facecolor='black',
                                   fill=True,
                                   lw=5))

            ax.add_patch(Rectangle((15 - width, 0), 2 * width, 7,
                                   edgecolor='black',
                                   facecolor='black',
                                   fill=True,
                                   lw=5))
            plt.grid(True)
            plt.xlim(0, 20)
            plt.ylim(0, 10)
            plt.savefig(output_dir + '/traj_generated_all_'+str(name)+'_'+str(successful_trials/len(initials))+'.pdf')

    return successful_trials/len(initials)













            # first = np.linspace(0,degree_poly,degree_poly+1)
            # second = np.multiply(first[1:-1],first[2::])
            # second = np.concatenate((np.zeros(2),second))
            # args_col = np.array([[np.linspace(1,degree_poly-1,degree_poly-1)] * (degree_poly-1)])[0,:,:]
            # args_row = np.array([[np.linspace(0, degree_poly-2, degree_poly-1)] * (degree_poly-1)])[0, :, :].T
            # div = np.ones ((degree_poly+1,degree_poly+1))
            # div[2::,2::] =  args_row+args_col
            # vec_time = get_poly_vector(total_time, degree_poly)
            # p_vector = np.multiply(second,vec_time)
            # P = np.outer(p_vector,p_vector)
            # P = np.divide(P,div)
            # P = P + 0.0001*np.eye(degree_poly+1)
            #
            # # Initial and final position with no velocity, and no acceleration
            # eval_0 = np.concatenate((np.ones(1),np.zeros(degree_poly)))
            # Ai = np.concatenate((eval_0,np.roll(eval_0,1),np.roll(eval_0, 2)))
            # Ai = Ai.reshape(3,degree_poly+1)
            # bix = np.array([start[0],0,0])
            # biy = np.array([start[1],0,0])
            #
            # vec = get_poly_vector(total_time, degree_poly)
            # Af = np.concatenate((vec, np.multiply(first,vec), np.multiply(vec,second)))
            # Af = Af.reshape(3,degree_poly+1)
            # bfx = np.array([goal[0],0,0])
            # bfy = np.array([goal[1],0,0])
            #
            # # Intermediate Points
            # Ax = np.array([])
            # Ay = np.array([])
            #
            # bx = np.array(intermediate_points)[:,0]
            # by = np.array(intermediate_points)[:,1]
            # for i, p in enumerate(intermediate_points):
            #     time = (i+1)/(len(intermediate_points)+2) * total_time
            #     vec = get_poly_vector(time,degree_poly)
            #     Ax = np.concatenate((Ax,vec))
            #     Ay = np.concatenate((Ay,vec))
            # Ax = Ax.reshape(int(len(Ay)/(degree_poly+1)),degree_poly+1)
            # Ay = Ax.reshape(int(len(Ay)/(degree_poly+1)),degree_poly+1)
            #
            # x = cp.Variable(degree_poly+1)
            # y = cp.Variable(degree_poly+1)
            # prob = cp.Problem(cp.Minimize((1 / 2) * cp.quad_form(x, P)+(1 / 2) * cp.quad_form(y, P) ),
            #                   [Ax @ x == bx,
            #                    Ay @ y ==  by,
            #                    Ai @ x == bix,
            #                    Af @ x == bfx,
            #                    Ai @ y == biy,
            #                    Af @ y == bfy])
            # prob.solve(verbose = True)
            #
            # labeled_times = np.linspace(0,total_time,n_train+2)
            # labeled_times = labeled_times[1:-1]
            #
            # labeled_pos_poly = [get_poly_vector(t,degree_poly) for t in labeled_times]
            # labeled_vel_poly = [np.multiply(first,np.roll(labeled_pos_poly_vec,1)) for labeled_pos_poly_vec in labeled_pos_poly]
            # labeled_acc_poly = [np.multiply(second,np.roll(labeled_pos_poly_vec,2)) for labeled_pos_poly_vec in labeled_pos_poly]
            #
            # labeled_points_x = np.array([ x.value @ poly for poly in labeled_pos_poly])
            # labeled_points_y = np.array([ y.value @ poly for poly in labeled_pos_poly])
            # labeled_vel_x = np.array([ x.value @ poly for poly in labeled_vel_poly])
            # labeled_vel_y = np.array([ y.value @ poly for poly in labeled_vel_poly])
            # labeled_acc_x = np.array([ x.value @ poly for poly in labeled_acc_poly])
            # labeled_acc_y = np.array([ y.value @ poly for poly in labeled_acc_poly])
            #
            #
            # X_lab = np.column_stack((labeled_points_x, labeled_points_y, labeled_vel_x, labeled_vel_y))
            # y_lab = np.column_stack((labeled_acc_x, labeled_acc_y))

            # If Random
            # X_unlab = np.array([])
            # weights = np.array([20,10,10,10]) # Max x, max y, max x_dot, max y_dot
            # for sample in range(n_unlab):
            #     s = np.multiply(np.random.uniform(0, 1, 4),weights)
            #     while not ((s[0] < 10-width or s[0] > 10 + width) or (s[1]>4 and s[1]<6)):
            #         s = np.multiply(np.random.uniform(0, 1, 4), weights)
            #         print(s)
            #     if len(X_unlab) == 0:
            #         X_unlab = s
            #     else:
            #         X_unlab = np.vstack((s,X_unlab))
            #         print(X_unlab.shape)
            # print(X_unlab.shape, X_lab.shape)



