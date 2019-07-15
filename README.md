# lisa-pf-noise

LISA Pathfinder noise analysis scripts

## Setup

1. Create a top-level `data/` directory.
2. Move run data into `data/` with the structure:
```
data/*run_name*
```

## PSD analysis

Execute `$ python3 psd.py 'run_a' 'run_b' ...`

## Spectral line analysis

Execute `$ python3 linechain.py 'run_a' 'run_b' ...`
