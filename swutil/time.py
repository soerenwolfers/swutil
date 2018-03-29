import timeit
from matplotlib import pyplot
from swutil.plots import plot_convergence

def snooze(value):
    '''
    time.sleep() substitute
    Keep busy for some time (very roughly and depending on machine value is ms)
    :param value: Time. Actual busy time depends on machine
    :type value: Number
    '''
    for i in range(int(0.65e3 * value)):
        __ = 2 ** (i / value)
    return 0

class Timer():
    def __enter__(self,name=''):
        self.name = name
        self.start_time = timeit.default_timer()
    def __exit__(self,*args):
        elapsed = timeit.default_timer()-self.start_time
        print('Elapsed time ({}): {.3f}s'.format(self.name,elapsed))

