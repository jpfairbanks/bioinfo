from __future__ import print_function
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools

def unpack(ntupl):
    return (ntupl[0][0], ntupl[0][1], ntupl[1])

prots = ['S', 'K', 'L', 'V']
media = ['elastin', 'gelatin']

trials = itertools.product(itertools.permutations(prots[:2]),media)
for ntupl in trials:
    print(ntupl)
    tupl = unpack(ntupl)
    print(tupl)
    #---- Load up some data ----#
    filename = 'data.%s.%s.%s.csv'%tupl
    title = 'cat%s eats cat%s on %s' % tupl
    df = pd.read_csv(filename)
    data_frame = df[['timevals', 'data']].set_index('timevals')
    model_frame = df[['time (min)', 'model']].set_index('time (min)')

    # ---- Show how the model compares to the actual data ----#
    joined = data_frame.join(model_frame, how='outer')
    joined.plot(style={'data':'k+', 'model':'-'}, title=title)
    plt.xlabel('time (minutes)')
    plt.ylabel('concentartion (M)')
    plt.savefig('plot.model.%s.%s.%s.png' % tupl)
    plt.figure()

    #----- Computing the residuals from the model to the data ----#
    # we need to fix the fact that experiment and model do not 
    # have the same index. So we fill and then subtract, then discard the
    # filled in values.
    diffs = joined.fillna().T.diff().T
    valid_indices = joined['data'].dropna().index
    resid = pd.Series(diffs.ix[valid_indices]['model'], name='residuals')
    resid.plot(title=title)
    plt.plot(np.arange(120), np.zeros(120), 'k-', )
    plt.legend(); 
    plt.xlabel('time (minutes)')
    plt.ylabel('concentartion (M)')
    plt.savefig('plot.resid.%s.%s.%s.png' % tupl)
    plt.show()
