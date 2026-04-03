import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

x = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 0]]
win = sliding_window_view(x, (2, 2))
print(win)