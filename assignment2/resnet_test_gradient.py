# As usual, a bit of setup
import numpy as np
import matplotlib.pyplot as plt
from cs231n.classifiers.cnn import *
from cs231n.data_utils import get_CIFAR10_data
from cs231n.gradient_check import eval_numerical_gradient_array, eval_numerical_gradient
from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.solver import Solver
import datetime
import warnings

# get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# for auto-reloading external modules
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

def rel_error(x, y):
  """ returns relative error """
  truncated_results = (np.abs(x) + np.abs(y)<1e-8).reshape(-1)
  print("--[rel_error]...[%d / %d] value too small to evalue" % (sum(truncated_results), len(truncated_results)))
  print("--[rel_error]... max absoluate diff %e" % np.max(np.abs(x-y)))
  rel_e = np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y)))
  return np.max(rel_e), np.mean(rel_e)


# %% Gradient check

num_inputs = 2
input_dim = (3, 32, 32)
reg = 0
num_classes = 10
np.random.seed(231)
X = np.random.randn(num_inputs, *input_dim)
y = np.random.randint(num_classes, size=num_inputs)

model = ResNet(input_dim = input_dim, weight_scale=1e-3, reg=reg, dtype=np.float64)
loss, grads = model.loss(X, y)
# Errors should be small, but correct implementations may have
# relative errors up to the order of e-2

f = lambda _: model.loss(X, y)[0]

for param_name in sorted(grads):
#for param_name in ['W0']:
    print('Evaluating %s... ' % (param_name))
    param_grad_num = eval_numerical_gradient(f, model.params[param_name], verbose=False, h=1e-7)
    param_grad_analy = grads[param_name];
    print('--[num_gradient]: max abs %e; min abs %e, mean %e, std %e' % (np.abs(param_grad_num).max(),np.abs(param_grad_num).min(),param_grad_num.mean(), param_grad_num.std()))
    print('--[ana_gradient]: max abs %e; min abs %e, mean %e, std %e' % (np.abs(param_grad_analy).max(),np.abs(param_grad_analy).min(),param_grad_analy.mean(), param_grad_analy.std()))
    e_max, e_mean = rel_error(param_grad_num, grads[param_name])
    print('--[rel_err]: max error %e; mean error %e' % (e_max, e_mean))
