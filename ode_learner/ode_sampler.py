import argparse
import numpy as np
import numpy.random as rand
import numpy.linalg as LA
import scipy 
import pandas as pd
import matplotlib.pyplot as plt
import itertools as itt

EPSILON = 0.0000001
def vprint(name,obj,verboseness=1):
    """Prints obj as a well formatted string

    :name: The name of the object to display
    :obj: The acutual variable name
    :returns: the formatted string

    """
    if args.verbose >= verboseness:
	string = "{0}:\n{1}".format(name,obj)
	print(string)
	return string
    return ""

def get_data(arg1):
    """Gets the data off the disk in order to optimize against it.

    :arg1: @todo
    :returns: @todo

    """
    return pd.load(YSEQ_PATH)

def get_mat(arg1):
    """@todo: Docstring for get_mat

    :arg1: @todo
    :returns: @todo

    """
    M = np.array([
           [ -1,  1,  0,  0,  0,],
           [ -1,  1,  1,  0,  0,],
           [  1, -1, -1,  0,  0,],
           [  0,  0,  0,  1,  0,],
           [  0,  0,  1,  0,  0,],
        ])
    return M

def orders_of_magnitue(arg1):
    """@todo: Docstring for orders_of_magnitue

    :arg1: @todo
    :returns: @todo

    """
    P = np.array([
           [ -1,  1,  0,  0,  0,],
           [ -1,  1,  1,  0,  0,],
           [  1, -1, -1,  0,  0,],
           [  0,  0,  0,  1,  0,],
           [  0,  0,  1,  0,  0,],
        ])
    return 10**P

def project(M, delta_t, num_t,):
    """ Project the differential equation given by M into the future

    :M: the matrix of coefficients
    :delta_t: the time step resolution
    :num_t: the number of time steps
    :returns: The sequence of states

    """
    yseq = np.ones((num_t+1, d))
    for i in range(num_t):
        yseq[i+1] = delta_t * M.T.dot(yseq[i]) + yseq[i]
        vprint('y', yseq[i+1], 2)

    return yseq

def generate_answer(delta_t, num_t, M):
    yseq = project(M, delta_t,NUM_T)
    #save the right answer and the fake data
    df = pd.DataFrame(yseq, index=xseq)
    df.save(YSEQ_PATH)
    MF = pd.DataFrame(M)
    MF.save("right_answer.df")
    if args.plot:
	df.plot()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="count")
    parser.add_argument("-p", "--plot", action="store_true")
    parser.add_argument("-g", "--gen", action="store_true")
    parser.add_argument("-s", "--solve", action="store_true")
    #parser.add_argument("files", nargs="+")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    d = 5
    NUM_T = 25
    MAX_T = 1.0
    YSEQ_PATH = "./yseq.df"
    eta = .05
    l = 2 # the order of the norm 1 or 2
    #M = np.matrix(np.zeros((d,d)))
    #M[0,1] = -.5
    #M[1,0] = .5
    M = get_mat(None)# * rand.random((d,d))
    vprint('M',M)
    xseq = np.linspace(0,MAX_T,NUM_T+1)
    delta_t = MAX_T/(NUM_T+1)
    if args.gen:
	generate_answer(delta_t, NUM_T, M)
    if args.solve:
        #find the best answer by optimization
	Y = get_data(None)
	M = get_mat(None) * rand.random((d,d))
	vprint('M_orig', M, verboseness=1)
	M_right = get_mat(None)
	frobs = [ ]
	errs  = [ ]
	iters=30
	i,j = 0,0
	for rd in range(iters):
	    for i,j in itt.product(range(d), range(d)):
		curr_Y = project(M, delta_t, NUM_T)
		err  = LA.norm((curr_Y - Y), l)
		errs.append(err)
		#check whether we should pertube up or down
		vprint('Y', curr_Y, 3)
		vprint('err',  err, 2)
		M[i,j] += eta 
		pos_Y = project(M, delta_t, NUM_T)
		vprint('+Y', pos_Y, 3)
		pos_err  = LA.norm((pos_Y - Y), l)
		vprint('+Yerr', pos_err, 2)
		M[i,j] -= 2*eta 
		neg_Y = project(M, delta_t, NUM_T)
		M[i,j] += eta 
		vprint('-Y', neg_Y, 3)
		neg_err  = LA.norm((neg_Y - Y), l)
		vprint('-Yerr', neg_err, 2)
		#choose the direction
		sign = np.argmin(np.array([neg_err, err, pos_err]))-1
		vprint('sign',sign, verboseness=3)
		M[i,j] += eta*sign
		#enforce sparsity
		M *= np.abs(M_right)
		#vprint('M_new', M, verboseness=1)
	    mat_err = LA.norm((M * M_right),'fro')
	    vprint('frob', mat_err, verboseness=2)
	    frobs.append(mat_err)
	vprint('M_prod', np.sqrt(M * M_right), verboseness=1)
