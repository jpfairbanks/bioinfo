import numpy as np
import numpy.random as rand
import scipy 
import pandas as pd
import matplotlib.pyplot as plt

def vprint(name,obj):
    """Prints obj as a well formatted string

    :name: @todo
    :obj: @todo
    :returns: the formatted string

    """
    string = "{0}:\n{1}".format(name,obj)
    print(string)
    return string

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

if __name__ == '__main__':
    d = 5
    NUM_T = 25
    MAX_T = 1.0
    #M = np.matrix(np.zeros((d,d)))
    #M[0,1] = -.5
    #M[1,0] = .5
    M = get_mat(None)# * rand.random((d,d))
    vprint('M',M)
    xseq = np.linspace(0,MAX_T,NUM_T+1)
    delta_t = MAX_T/(NUM_T+1)
    yseq = np.ones((NUM_T+1,d))
    for i in range(NUM_T):
        yseq[i+1] = delta_t*M.T.dot(yseq[i]) + yseq[i]
        vprint('y',yseq[i+1])
    df = pd.DataFrame(yseq, index=xseq)
    df.plot()
    #plt.plot(xseq,yseq,)
    M = M+np.eye(d)
    plt.plot(xseq,yseq)

