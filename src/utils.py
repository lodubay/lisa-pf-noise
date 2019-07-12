import sys

class Progress:
    '''A loop progress indicator class'''
    def __init__(self, message=''):
        # Set up the progress indicator with a status message
        self.progress = 0
        sys.stdout.write(f'{message}   0%\b')
        sys.stdout.flush()
        
    def update(self, i, iterable):
        # Call inside loop to update progress indicator
        self.progress = int((i+1) / len(iterable) * 100)
        pstr = str(self.progress)
        sys.stdout.write('\b' * len(pstr) + pstr)
        sys.stdout.flush()
        if self.progress == 100: self.end()

    def end(self):
        # Terminate progress indicator
        sys.stdout.write('\n')
        sys.stdout.flush()
    
