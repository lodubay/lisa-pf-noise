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

class Log:
    '''A class for outputting to a log file'''
    def __init__(self, log_file=None, header='Log'):
        self.log_file = log_file
        if log_file:
            print(f'Logging output to {log_file}')
            with open(log_file, 'w+') as f:
                f.write(header)
                f.write('\n\n')
    
    def log(self, message=''):
        if self.log_file:
            with open(self.log_file, 'a+') as f:
                f.write(message)
                f.write('\n')
        
