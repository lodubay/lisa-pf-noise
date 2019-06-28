#import time_functions as tf
#import psd
import plot
import linechain as lc
import pandas as pd
import numpy as np
import os
from pymc3.stats import hpd
import matplotlib.pyplot as plt

run = 'ltp_run_b1'
output_dir = os.path.join('out', run)
if not os.path.exists(output_dir): os.makedirs(output_dir)
model_file = os.path.join(output_dir, run + '_line_evidence.dat')
time = 1143789532
channel = 4
model = lc.best_line_model(run, time, channel)
lc_params = lc.get_model_params(run, time, channel, model)
#lc_hpd = hpd(lc_params.to_numpy(), alpha=0.05)
#print(lc_hpd)
#med = lc_params.median()
#std = lc_params.std()
#print(med)
#lower = med - std
#upper = med + 5 * std
#print(lower)
#print(lc_params[lc_params.all() > lower.all() and lc_params.all() < upper.all()])
#print(lc.get_param_centroids(lc_params, model))
#plot.line_params_corner(lc_params)
plot.chain_consumer(lc_params)
plot.line_params_corner(lc_params)
