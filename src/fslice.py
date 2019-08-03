#!/usr/bin/env python3

import os
import argparse
from glob import glob
import configparser

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr

import utils

# Font parameters
fig_title_size = 28
subplot_title_size = 24
legend_label_size = 20
ax_label_size = 20
tick_label_size = 18
offset_size = 16

# Tick parameters
major_tick_length = 10
minor_tick_length = 5

# Other parameters
subplot_title_pad = 15
spine_pad = 10 # Spine offset from subplots
ax_label_pad = 10 + ax_label_size # Padding between axis label and spine
scaled_offset = 0.0025 * spine_pad # Scaled spine_pad

def main():
    # Argument parser
    args = utils.add_parser('Frequency slice analysis.')

    # Plot config parser
    config = configparser.ConfigParser()
    config.read('src/plotconfig.ini')
    
    # Initialize run objects; skip missing directories
    runs = utils.init_runs(args.runs)
    
    # Import impacts file, if any
    impacts_file = 'impacts.dat'
    impacts = np.array([])
    if os.path.exists(impacts_file):
        print('Importing impacts file...')
        impacts = utils.get_impacts(impacts_file)
    
    # Frequencies to examine
    frequencies = np.array([1e-3, 3e-3, 5e-3, 1e-2, 3e-2, 5e-2])
    
    # Individual plots
    for run in runs:
        print(f'\n-- {run.mode} {run.name} --')
        # Import PSD summaries
        df = pd.read_pickle(run.psd_file)
        
        p = utils.Progress(run.channels, 'Plotting frequency slices...')
        for c, channel in enumerate(run.channels):
            # Plot grid of time series
            fig = utils.gridplot(fslice, df, channel, frequencies,
                    f'{run.mode.upper()} channel {channel}\n' + \
                            'Power at selected frequencies',
                    f'Days elapsed since\n{run.start_date} UTC', 'Power', 
                    config)
            
            # Save plot
            plot_file = os.path.join(run.plot_dir, f'fslice{c}.png')
            plt.savefig(plot_file, bbox_inches='tight')
            plt.close()
            
            # Update progress
            p.update(c)
    
    # Comparison plots
    if args.compare:
        # Output directory
        multirun_dir = os.path.join('out', 'multirun')
        if not os.path.exists(multirun_dir): 
            os.makedirs(multirun_dir)
        plot_dir = os.path.join(multirun_dir, 'plots')
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        
        run_str = [f'{run.mode} {run.name}' for run in runs]
        print(f'\n-- {", ".join(run_str)} --')
        
        # One plot per channel
        p = utils.Progress(runs[0].channels, 
                'Plotting frequency slice comparisons...')
        for c, channel in enumerate(runs[0].channels):
            # Set up figure, grid
            fig = plt.figure(figsize=(8 + len(runs) * 4, 12))
            plot_height = 2 # Relative height of each subplot
            hspace = 2 # Relative vertical spaceing between subplots
            grid = plt.GridSpec(plot_height * len(frequencies), len(runs), hspace=2, 
                    wspace=0.25)
            fig.suptitle(
                    f'Channel {channel}\nPower at selected frequencies over time', 
                    fontsize=fig_title_size)
            fig.tight_layout()
        
            # Plot one column for each run
            for j, run in enumerate(runs):
                # Import PSD summaries
                df = pd.read_pickle(run.psd_file)
                # Plot highest frequency on top
                frequencies = np.flip(np.sort(frequencies))
                
                # Plot one row for each frequency
                for i, freq in enumerate(frequencies):
                    # Get exact frequency along which to slice
                    freq = utils.get_exact_freq(df, freq)
                    freq_text = f'%s mHz' % float('%.3g' % (freq * 1000.))
                    
                    # Set up subplot
                    ax = fig.add_subplot(
                            grid[plot_height*i:plot_height*i+plot_height, j])
                    
                    # Plot frequency slice
                    fslice(fig, ax, df, run, channel, freq)
                    
                    # Subplot title if top plot
                    if i == 0:
                        ax.set_title(f'{run.mode.upper()}', 
                                fontsize=subplot_title_size,
                                pad=subplot_title_pad)
                    
                    # More mathy exponent label
                    ax.ticklabel_format(axis='y', useMathText=True)
                    exp_txt = ax.yaxis.get_offset_text()
                    exp_txt.set_x(-0.005 * spine_pad)
                    exp_txt.set_size(offset_size)
                
                    # Format left vertical axis
                    ylim = ax.get_ylim()
                    ax.set_ylabel(freq_text, fontsize=tick_label_size)
                    y_formatter = tkr.ScalarFormatter(useOffset=False)
                    ax.yaxis.set_major_formatter(y_formatter)
                    ax.yaxis.set_minor_locator(tkr.AutoMinorLocator())
                    ax.spines['left'].set_bounds(ylim[0], ylim[1])
                    ax.spines['left'].set_position(('outward', spine_pad))
                    ax.tick_params(axis='y', which='major', 
                            labelsize=tick_label_size)
                    ax.tick_params(axis='both', which='major', 
                            length=major_tick_length)
                    ax.tick_params(axis='both', which='minor', 
                            length=minor_tick_length)
                
                    # Format bottom horizontal axis
                    xlim = ax.get_xlim()
                    ax.spines['bottom'].set_bounds(xlim[0], xlim[1])
                    if i+1 < len(frequencies):
                        # Remove bottom axis if not the bottom plot
                        ax.spines['bottom'].set_visible(False)
                        ax.tick_params(bottom=False)
                        # Horizontal axis ticks
                        ax.xaxis.set_major_locator(tkr.NullLocator())
                        ax.xaxis.set_minor_locator(tkr.NullLocator())
                    else:
                        # Horizontal axis for bottom plot
                        ax.set_xlabel(f'Days elapsed since\n{run.start_date} UTC', 
                                fontsize=ax_label_size)
                        ax.spines['bottom'].set_visible(True)
                        ax.spines['bottom'].set_position(('outward', spine_pad))
                        ax.tick_params(bottom=True)
                        ax.tick_params(axis='x', which='major', 
                                labelsize=tick_label_size)
                        # Minor ticks
                        ax.xaxis.set_minor_locator(tkr.AutoMinorLocator())
                        
                        # Find micrometeoroid impacts, if any
                        if len(impacts) > 0:
                            # Find impact times
                            gps_times = df.index.unique(level='TIME')
                            days = run.gps2day(gps_times)
                            impact_times = impacts[
                                    (impacts['GPS'] >= gps_times[0]) \
                                  & (impacts['GPS'] <= gps_times[-1])]
                            impact_days = run.gps2day(impact_times['GPS']).to_numpy()
                        
                        # Plot micrometeoroid impacts, if any
                        impact_plt = ax.scatter(impact_days, 
                                [ylim[0] - (ylim[1]-ylim[0]) * 0.013 * spine_pad] \
                                        * len(impact_days), 
                                c='red', marker='x', label='Impact event', 
                                clip_on=False, s=100)
                
                    # Remove spines and ticks for other axes
                    ax.spines['right'].set_visible(False)
                    ax.spines['top'].set_visible(False)
            
            # Get legend handles
            handles, labels = ax.get_legend_handles_labels()

            # Add big subplot for common y axis label
            ax = fig.add_subplot(1, 1, 1, frameon=False)
            ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, 
                    right=False, which='both')
            ax.grid(False)
            # y axis label
            ax.set_ylabel('Power at selected frequency', 
                    fontsize=subplot_title_size, ha='center', va='center', 
                    labelpad=ax_label_pad * 2 + 8)
        
            # Make legend
            handles += [impact_plt]
            order = [0, 2, 1, 3] # Reorder legend
            fig.legend([handles[i] for i in order], 
                    [labels[i] for i in order],
                    fontsize=legend_label_size, 
                    loc='upper right', bbox_to_anchor=(1.05, 1), 
                    bbox_transform=plt.gcf().transFigure)
            plt.subplots_adjust(top=1 - (4 * 0.0020 * legend_label_size))
        
            # Save plot
            plot_file = os.path.join(plot_dir, f'fslice{c}.png')
            plt.savefig(plot_file, bbox_inches='tight')
            plt.close()
            
            # Update progress
            p.update(c)

def fslice(fig, ax, df, channel, frequency):
    '''
    Plots power at a specific frequency over time.

    Input
    -----
      fig, ax : the figure and axes of the plot
      df : DataFrame, the full summary output from import_psd.py
      run : Run object
      channel : str, the channel to investigate
      frequency : float, the approximate frequency along which to slice the PSD
    '''
    # Get exact frequency along which to slice
    frequency = utils.get_exact_freq(df, frequency)
    freq_text = f'%s mHz' % float('%.3g' % (frequency * 1000.))

    # Isolate given channel
    time_series = df.loc[channel].xs(frequency, level='FREQ')
    days_elapsed = utils.gps2day(time_series.index) # Convert to days elapsed
    
    # Plot 90% credible interval
    ax.fill_between(days_elapsed, time_series['CI_90_LO'], 
            time_series['CI_90_HI'],
            color='#b3cde3', label='90% credible interval')
    # Plot 50% credible interval
    ax.fill_between(days_elapsed, time_series['CI_50_LO'], 
            time_series['CI_50_HI'], 
            color='#8c96c6', label='50% credible interval')
    # Plot median
    ax.plot(days_elapsed, time_series['MEDIAN'], 
            label='Median', color='#88419d')

    # Axis title
    ax.set_title(freq_text)
            
    # Vertical axis limits
    med = time_series['MEDIAN'].median()
    std = time_series['MEDIAN'].std()
    hi = min(2 * (time_series['CI_90_HI'].quantile(0.95) - med), 
             max(time_series['CI_90_HI']) - med)
    lo = min(2 * abs(med - time_series['CI_90_LO'].quantile(0.05)), 
             abs(med - min(time_series['CI_90_LO'])))
    ylim = (med - lo, med + hi)
    ax.set_ylim(ylim)
    
    # Horizontal axis limits
    ax.set_xlim((min(days_elapsed), max(days_elapsed)))

if __name__ == '__main__':
    main()

