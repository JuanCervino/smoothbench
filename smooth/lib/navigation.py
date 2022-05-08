import numpy as np
import cvxpy as cp
import pickle,os

def get_poly_vector(t,degree):
    vec = np.ones(degree+1)
    aux = t
    vec[1] = t
    for d in range(2,degree+1):
        aux = aux * t
        vec[d] = aux

    return vec


def create_dataset(dataset, n_dim, n_train, n_unlab, width, data_dir):

    assert dataset in ['window']
    file_name = '/'+dataset+'_train'+str(n_train)+'_unlab'+str(n_unlab)+'_noise'+str(width)+'.p'



    if os.path.exists(data_dir+file_name):
        # [X_lab,y_lab,X_unlab, y_unlab] = pickle.load(data_dir+file_name)
        with open(data_dir+file_name, 'rb') as pickle_file:
            [X_lab,y_lab,X_unlab, y_unlab] = pickle.load(pickle_file)
    else:
        if dataset == 'window':
            goal = [19,1]
            start = [1,1]
            # intermediate_points = [[5,3],[10,5],[15,3]]
            intermediate_points = [[5,3],[10,5]]

            total_time = 1

            degree_poly = 10
            first = np.linspace(0,degree_poly,degree_poly+1)
            second = np.multiply(first[1:-1],first[2::])
            second = np.concatenate((np.zeros(2),second))
            args_col = np.array([[np.linspace(1,degree_poly-1,degree_poly-1)] * (degree_poly-1)])[0,:,:]
            args_row = np.array([[np.linspace(0, degree_poly-2, degree_poly-1)] * (degree_poly-1)])[0, :, :].T
            div = np.ones ((degree_poly+1,degree_poly+1))
            div[2::,2::] =  args_row+args_col

            P = np.outer(second,second)
            P = np.divide(P,div)
            P = P + 0.0001*np.eye(degree_poly+1)
            # Initial and final position with no velocity, and no acceleration
            eval_0 = np.concatenate((np.ones(1),np.zeros(degree_poly)))
            Ai = np.concatenate((eval_0,np.roll(eval_0,1),np.roll(eval_0, 2)))
            Ai = Ai.reshape(3,degree_poly+1)
            bix = np.array([start[0],0,0])
            biy = np.array([start[1],0,0])

            Af = np.concatenate((np.ones(degree_poly+1), first, second))
            Af = Af.reshape(3,degree_poly+1)
            bfx = np.array([goal[0],0,0])
            bfy = np.array([goal[1],0,0])

            # Intermediate Points
            Ax = np.array([])
            Ay = np.array([])

            bx = np.array(intermediate_points)[:,0]
            by = np.array(intermediate_points)[:,1]
            for i, p in enumerate(intermediate_points):
                time = (i+1)/(len(intermediate_points)+2)
                vec = get_poly_vector(time,degree_poly)
                Ax = np.concatenate((Ax,vec))
                Ay = np.concatenate((Ay,vec))
            Ax = Ax.reshape(int(len(Ay)/(degree_poly+1)),degree_poly+1)
            Ay = Ax.reshape(int(len(Ay)/(degree_poly+1)),degree_poly+1)

            x = cp.Variable(degree_poly+1)
            y = cp.Variable(degree_poly+1)
            prob = cp.Problem(cp.Minimize((1 / 2) * cp.quad_form(x, P)+(1 / 2) * cp.quad_form(y, P) ),
                              [Ax @ x == bx,
                               Ay @ y ==  by,
                               Ai @ x == bix,
                               Af @ x == bfx,
                               Ai @ y == biy,
                               Af @ y == bfy])
            prob.solve()

            labeled_times = np.linspace(0,1,200)
            labeled_poly = [get_poly_vector(t,degree_poly) for t in labeled_times]
            print(labeled_poly, x.value)
            labeled_points_x = [ x.value @ poly for poly in labeled_poly]
            labeled_points_y = [ y.value @ poly for poly in labeled_poly]
            X_lab = [labeled_points_x,labeled_points_y]
    return [X_lab,None,None,None]