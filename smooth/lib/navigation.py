import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cp
import pickle,os
import scipy
from matplotlib.patches import Rectangle


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

def create_dataset(dataset, n_dim, n_train, n_unlab, width, data_dir):

    assert dataset in ['window','center','Dijkstra_grid_window']
    file_name = '/'+dataset+'_train'+str(n_train)+'_unlab'+str(n_unlab)+'_noise'+str(width)+'.p'

    total_time = 4

    if os.path.exists(data_dir+file_name):
        # [X_lab,y_lab,X_unlab, y_unlab] = pickle.load(data_dir+file_name)
        with open(data_dir+file_name, 'rb') as pickle_file:
            [X_lab,y_lab,X_unlab, y_unlab] = pickle.load(pickle_file)
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
            weights = np.array([20,10]) # Max x, max y, max x_dot, max y_dot
            x = np.linspace(0, 20, 2*n_train-1)
            y = np.linspace(0, 10, n_train)
            # full coorindate arrays
            xx, yy = np.meshgrid(x, y)
            graph = np.column_stack((xx.flatten(),yy.flatten()))
            graph = np.vstack((graph,np.array([19,1])))
            list_of_delete = []
            print(graph.shape)

            for sample in range(graph.shape[0]):
               if not ((graph[sample,0] < 10-width or graph[sample,0]  > 10 + width) or (graph[sample,1] >4 and graph[sample,1] <6)):
                   list_of_delete = list_of_delete + [sample]

            graph = np.delete(graph,list_of_delete,axis=0)
            matrix_graph = scipy.spatial.distance.cdist(graph,graph)
            matrix_graph = np.where(matrix_graph > 0.01+np.sqrt((10/(n_train-1))**2+(10/(n_train-1))**2), 0, matrix_graph)
            # matrix_graph = np.where(matrix_graph > 2.5, 0, matrix_graph)

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
            X_lab = np.array(graph[path[:-1],:])
            y_lab = np.array(graph[path[1:]]-graph[path[:-1]])/total_time
            X_unlab = None
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



        return [X_lab, y_lab, X_unlab, y_unlab]




















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



