from __future__ import print_function
import os.path as path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools

def tuple_repr(t):
    """Format a tuple to go into a filename with periods instead of commas
       Ends in a period.
    :t: the tuple to take
    :returns: item0.item1...

    """
    perf = lambda i: str(i)+'.'
    string = ''.join(map(perf, t))
    return string

prots = ['S', 'K', 'L', 'V']
media = ['elastin', 'gelatin']

length = 120
titlefmt    = 'cat%s on %s'
filenamefmt = 'data.%scsv'
files = ['data.K.S.elastin.csv']
for fname in files:
    print(fname)
    bname = path.basename(fname)
    components = bname.split('.')
    #if components[-1] is not 'csv':
    #throw away the prefix and the fileextension
    tupl = components[1:-1]
    print(tupl)
    #---- Load up some data ----#
    title = titlefmt % (str(tupl[:-1]), tupl[-1])
    df = pd.read_csv(fname)
    data_frame = df[['timevals', 'data']].set_index('timevals')
    model_frame = df[['time (min)', 'model']].set_index('time (min)')

    # ---- Show how the model compares to the actual data ----#
    joined = data_frame.join(model_frame, how='outer')
    joined.plot(style={'data':'ko', 'model':'-'}, title=title)
    plt.xlabel('time (minutes)')
    plt.ylabel('concentartion (M)')
    plt.savefig('plot.model.%spng' % tuple_repr(tupl))

    #----- Computing the residuals from the model to the data ----#
    # we need to fix the fact that experiment and model do not 
    # have the same index. So we fill and then subtract, then discard the
    # filled in values.
    plt.figure()
    diffs = joined.fillna().T.diff().T
    valid_indices = joined['data'].dropna().index
    resid = pd.Series(diffs.ix[valid_indices]['model'], name='residuals')
    resid.plot(kind='bar', title=title)
    plt.plot(np.arange(length), np.zeros(length), 'k-', )
    plt.legend(); 
    plt.xlabel('time (minutes)')
    plt.ylabel('concentartion (M)')
    plt.savefig('plot.resid.%spng' % tuple_repr(tupl))
    plt.show()
