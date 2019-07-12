import sys

class Progress:
    '''A loop progress indicator class'''
    def __init__(self, iterable, message=''):
        # Set up the progress indicator with a status message
        self.progress = 0
        self.steps = len(iterable)
        sys.stdout.write(f'{message}   {self.progress}%\b')
        sys.stdout.flush()
        
    def update(self, i):
        # Call inside loop to update progress indicator
        self.progress = int((i+1) / self.steps * 100)
        prog_str = str(self.progress)
        sys.stdout.write('\b' * len(prog_str) + prog_str)
        sys.stdout.flush()
        if self.progress == 100: self.end()

    def end(self):
        # Terminate progress indicator; automatically called by self.update()
        sys.stdout.write('\n')
        sys.stdout.flush()
    
