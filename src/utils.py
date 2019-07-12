import sys

def progress(iterable, func, message):
    sys.stdout.write('{func}   0%\b')
    sys.stdout.flush()
    
