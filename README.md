# lisa-pf-noise

LISA Pathfinder noise analysis scripts

## Setup

1. Create a top-level `data/` directory.
2. Each run should be organized into one directory. The name of the run is the 
   name of the top directory, and each subdirectory contains data for a certain
   time. Move the run data into `data/`. Example structure:
```
data/
    run_k/
        run_k_1159724317/
        run_k_1159725956/
        run_k_1159727594/
        ...
    ltp_run_b/
        run_b_1143763317/
        run_b_1143764956/
        run_b_1143766594/
        ...
```
3. Each subdirectory (i.e., time directory) must contain the following files:
```
linechain_channel0.dat
linechain_channel1.dat
linechain_channel2.dat
linechain_channel3.dat
linechain_channel4.dat
linechain_channel5.dat
psd.dat.0
psd.dat.1
psd.dat.2
...
psd.dat.99
```

## PSD analysis

Execute `$ src/psd.py 'run_a' 'run_b' ...`

The script will first generate and save summary DataFrames, and then create
colormap, time slice, and frequency slice plots for each channel. Summaries
are saved to `out/<run_name>/summaries/` and plots are
saved to `out/<run_name>/psd_plots/`.

If the script is run multiple times on the same run, it will ask whether to 
generate new summaries (which takes a long time) or use the existing summary
data. Pass the `--overwrite-all` option to re-generate all summary files without
asking, or use the `--keep-all` option to use all existing summaries and just 
generate new plots.

If no arguments are specified, the script will run on all runs within the
`data/` directory.

## Spectral line analysis

Execute `$ src/linechain.py 'run_a' 'run_b' ...`

This script looks at the time variation of spectral line parameters. It draws
from the `linechain_channel<X>.dat` files and finds the median value and
credible intervals for each time and channel. The script outputs a summary
DataFrame, then plots each parameter over time. Summaries
are saved to `out/<run_name>/summaries/` and plots are
saved to `out/<run_name>/linechain_plots/`.

Command line arguments are the same as for the PSD analysis.

