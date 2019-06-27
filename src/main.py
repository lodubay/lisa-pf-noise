#import time_functions as tf
#import psd
import plot
import linechain as lc
import pandas as pd
import numpy as np
import os

run = 'ltp_run_b1'
output_dir = os.path.join('out', run)
if not os.path.exists(output_dir): os.makedirs(output_dir)
model_file = os.path.join(output_dir, run + '_line_evidence.dat')
#for channel in range(6):
#    best_model = []
#    for time in lc.get_lines(run, channel, model_file):
#        best_model.append(lc.best_line_model(run, time, channel))
#    print('Best model for channel ' + str(channel) + ': ' + str(best_model))
plot.line_params(lc.get_line_params(run, 1143789532, 4))
