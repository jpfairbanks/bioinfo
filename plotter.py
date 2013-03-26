from __future__ import print_function
import math
import os.path as path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools
import argparse

def tuple_repr(t):
    """Format a tuple to go into a filename with periods instead of commas
       Ends in a period.
    :t: the tuple to take
    :returns: item0.item1...

    """
    perf = lambda i: str(i)+'.'
    string = ''.join(map(perf, t))
    return string

def process_file(fname):
    """Take a filename and make the plot and residual plot"""
    if args.verbose:
        print(fname)
    bname = path.basename(fname)
    components = bname.split('.')
    #if components[-1] is not 'csv':
    #throw away the prefix and the fileextension
    tupl = components[1:-1]
    if args.verbose:
        print(tupl)
    #---- Load up some data ----#
    title = titlefmt % (str(tupl[:-1]), tupl[-1])
    df = pd.read_csv(fname)
    data_col = df.columns[1]
    if args.verbose:
        print(df)
	print(df.head())
    data_frame = df[df.columns[0:2]].set_index('timevals')
    model_frame = df[df.columns[4:6]].set_index('time (min)')
    if args.verbose:
	print(data_frame.head(10))
	print(model_frame.head(10))

    # ---- Show how the model compares to the actual data ----#
    joined = data_frame.join(model_frame, how='outer')
    if args.verbose:
	print(joined.head(10))
    joined.plot(style={data_col:'ko', 'model':'-'}, title=title)
    plt.xlabel('time (minutes)')
    plt.ylabel('concentartion (M)')
    plt.savefig('plot.model.%spng' % tuple_repr(tupl))

    #----- Computing the residuals from the model to the data ----#
    # we need to fix the fact that experiment and model do not 
    # have the same index. So we fill and then subtract, then discard the
    # filled in values.
    if args.residuals:
	if args.verbose:
	    print('starting residuals')
        plt.figure()
        diffs = joined.fillna().T.diff().T
        valid_indices = joined[data_col].dropna().index
	if args.verbose:
	    print(valid_indices[-1])
	    print(type(valid_indices[-1]))
	if math.isnan(valid_indices[-1]):
	    print('we hit a nan')
	    valid_indices = valid_indices[0:-1]
	    print(valid_indices) 
        resid = pd.Series(diffs.ix[valid_indices]['model'], name='residuals')
        resid.plot(kind='bar', title=title)
        plt.plot(np.arange(length), np.zeros(length), 'k-', )
        plt.legend(); 
        plt.xlabel('time (minutes)')
        plt.ylabel('concentartion (M)')
        plt.savefig('plot.resid.%spng' % tuple_repr(tupl))
        plt.show()
    return df

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("-r", "--residuals", action="store_true")
    parser.add_argument("files", nargs="+")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    prots = ['S', 'K', 'L', 'V']
    media = ['elastin', 'gelatin']

    length = 120
    titlefmt    = 'cat%s on %s'
    filenamefmt = 'data.%scsv'
    #files = ['data.K.S.elastin.csv']
    for fname in args.files:
	try:
            df = process_file(fname)
	except Exception as e:
	    print('we failed to process file: %s'%fname)
	    print(e)
