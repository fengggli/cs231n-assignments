
# coding: utf-8

# # Convolutional Networks
# So far we have worked with deep fully-connected networks, using them to explore different optimization strategies and network architectures. Fully-connected networks are a good testbed for experimentation because they are very computationally efficient, but in practice all state-of-the-art results use convolutional networks instead.
# 
# First you will implement several layer types that are used in convolutional networks. You will then use these layers to train a convolutional network on the CIFAR-10 dataset.

# In[ ]:


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

get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# for auto-reloading external modules
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))




# %% 
# Rel errors should be around e-9 or less
# from cs231n.fast_layers import conv_forward_fast, conv_backward_fast
from time import time
np.random.seed(231)
x = np.random.randn(100, 3, 31, 31)
w = np.random.randn(25, 3, 3, 3)
b = np.random.randn(25,)
dout = np.random.randn(100, 25, 16, 16)
conv_param = {'stride': 2, 'pad': 1}

t0 = time()
out_fast, cache_fast = conv_forward_fast(x, w, b, conv_param)
t1 = time()
dx_fast, dw_fast, db_fast = conv_backward_fast(dout, cache_fast)
t2 = time()

print('\nTesting conv__fast:')
print('Forward: %fs' % (t1 - t0))
print('Backward: %fs' % (t2 - t1))



# %%

# %% check implementations
from cs231n.fast_layers import conv_forward_fast, conv_backward_fast
from time import time

from cs231n.layer_utils import conv_relu_pool_forward, conv_relu_pool_backward
np.random.seed(231)
x = np.random.randn(2, 3, 16, 16)
w = np.random.randn(3, 3, 3, 3)
b = np.random.randn(3,)
dout = np.random.randn(2, 3, 8, 8)
conv_param = {'stride': 1, 'pad': 1}
pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

out, cache = conv_relu_pool_forward(x, w, b, conv_param, pool_param)
dx, dw, db = conv_relu_pool_backward(dout, cache)

dx_num = eval_numerical_gradient_array(lambda x: conv_relu_pool_forward(x, w, b, conv_param, pool_param)[0], x, dout)
dw_num = eval_numerical_gradient_array(lambda w: conv_relu_pool_forward(x, w, b, conv_param, pool_param)[0], w, dout)
db_num = eval_numerical_gradient_array(lambda b: conv_relu_pool_forward(x, w, b, conv_param, pool_param)[0], b, dout)

# Relative errors should be around e-8 or less
print('Testing conv_relu_pool')
print('dx error: ', rel_error(dx_num, dx))
print('dw error: ', rel_error(dw_num, dw))
print('db error: ', rel_error(db_num, db))

# %% verify plus
np.random.seed(231)

x1 = np.random.randn(2, 3, 8, 8)
x2 = np.random.randn(2, 3, 8, 8)
dout = np.random.randn(2, 3, 8, 8)

out, cache = plus_forward(x1, x2)
dx1, dx2 = plus_backward(dout, cache)

dx1_num = eval_numerical_gradient_array(lambda x1: plus_forward(x1, x2)[0], x1, dout)
dx2_num = eval_numerical_gradient_array(lambda x2: plus_forward(x1, x2)[0], x2, dout)

# Relative errors should be around e-8 or less
print('Testing plus forward:')
print('dx1 error: ', rel_error(dx1_num, dx1))
print('dx1 error: ', rel_error(dx2_num, dx2))

# In[ ]:


from cs231n.layer_utils import conv_iden_relu_forward, conv_iden_relu_backward
np.random.seed(231)
x = np.random.randn(2, 3, 8, 8)
iden = np.random.randn(2, 3, 8, 8)
w = np.random.randn(3, 3, 3, 3)
b = np.random.randn(3,)
dout = np.random.randn(2, 3, 8, 8)
conv_param = {'stride': 1, 'pad': 1}

out, cache = conv_iden_relu_forward(x, iden, w, b, conv_param)
dx, diden, dw, db = conv_iden_relu_backward(dout, cache)

dx_num = eval_numerical_gradient_array(lambda x: conv_iden_relu_forward(x, iden, w, b, conv_param)[0], x, dout)
diden_num = eval_numerical_gradient_array(lambda iden: conv_iden_relu_forward(x, iden, w, b, conv_param)[0], iden, dout)
dw_num = eval_numerical_gradient_array(lambda w: conv_iden_relu_forward(x, iden, w, b, conv_param)[0], w, dout)
db_num = eval_numerical_gradient_array(lambda b: conv_iden_relu_forward(x, iden, w, b, conv_param)[0], b, dout)

# Relative errors should be around e-8 or less
print('Testing conv_relu:')
print('dx error: ', rel_error(dx_num, dx))
print('dx error: ', rel_error(diden_num, diden))
print('dw error: ', rel_error(dw_num, dw))
print('db error: ', rel_error(db_num, db))


# %% test resisual block

np.random.seed(231)
x = np.random.randn(2, 3, 8, 8)
iden = np.random.randn(2, 3, 8, 8)
w1 = np.random.randn(3, 3, 3, 3)
b1 = np.random.randn(3,)

w2 = np.random.randn(3, 3, 3, 3)
b2 = np.random.randn(3,)

dout = np.random.randn(2, 3, 8, 8)
conv_param = {'stride': 1, 'pad': 1}


out, cache = resnet_basic_no_bn_forward(x, w1, b1, w2, b2, conv_param)
dx, dw1, db1, dw2, db2 = resnet_basic_no_bn_backward(dout, cache)

dx_num = eval_numerical_gradient_array(lambda x: resnet_basic_no_bn_forward(x, w1, b1, w2, b2, conv_param)[0], x, dout)
dw1_num = eval_numerical_gradient_array(lambda w1: resnet_basic_no_bn_forward(x, w1, b1, w2, b2, conv_param)[0], w1, dout)
db1_num = eval_numerical_gradient_array(lambda b1: resnet_basic_no_bn_forward(x, w1, b1, w2, b2, conv_param)[0], b1, dout)
dw2_num = eval_numerical_gradient_array(lambda w2: resnet_basic_no_bn_forward(x, w1, b1, w2, b2, conv_param)[0], w2, dout)
db2_num = eval_numerical_gradient_array(lambda b2: resnet_basic_no_bn_forward(x, w1, b1, w2, b2, conv_param)[0], b2, dout)

# Relative errors should be around e-8 or less
print('Testing conv_relu:')
print('dx error: ', rel_error(dx_num, dx))
print('dw1 error: ', rel_error(dw1_num, dw1))
print('db1 error: ', rel_error(db1_num, db1))
print('dw2 error: ', rel_error(dw2_num, dw2))
print('db2 error: ', rel_error(db2_num, db2))


# %% sanity check loss
# After you build a new network, one of the first things you should do is sanity check the loss. When we use the softmax loss, we expect the loss for random weights (and no regularization) to be about `log(C)` for `C` classes. When we add regularization this should go up.


model = ResNet()

N = 50
X = np.random.randn(N, 3, 32, 32)
y = np.random.randint(10, size=N)

loss, grads = model.loss(X, y)
print('Initial loss (no regularization): ', loss)

model.reg = 0.5
loss, grads = model.loss(X, y)
print('Initial loss (with regularization): ', loss)

# %% Gradient check
# After the loss looks reasonable, use numeric gradient checking to make sure that your backward pass is correct. When you use numeric gradient checking you should use a small amount of artifical data and a small number of neurons at each layer. Note: correct implementations may still have relative errors up to the order of e-2.

# relative errors I got: (W0: 8.36e-5), (W1_0_0_0, 4.08e-2), (W1_0_1: 3.56e-2),(W3: 1.2e-7) (b0:2.29e-6), (b1_0_0, 747e-1), (b1_0_1: 3.98 e-6), (b3: 1.23e-9)
num_inputs = 2
input_dim = (3, 32, 32)
reg = 0.0
num_classes = 10
np.random.seed(231)
X = np.random.randn(num_inputs, *input_dim)
y = np.random.randint(num_classes, size=num_inputs)

model = ResNet(input_dim = input_dim,  dtype=np.float64)
loss, grads = model.loss(X, y)
# Errors should be small, but correct implementations may have
# relative errors up to the order of e-2
for param_name in sorted(grads):
    f = lambda _: model.loss(X, y)[0]
    param_grad_num = eval_numerical_gradient(f, model.params[param_name], verbose=False, h=1e-7)
    e = rel_error(param_grad_num, grads[param_name])
    print('%s max relative error: %e' % (param_name, rel_error(param_grad_num, grads[param_name])))


# %% cifar10
# Load the (preprocessed) CIFAR10 data.

data = get_CIFAR10_data()
for k, v in data.items():
  print('%s: ' % k, v.shape)


# %%  Overfit small data

np.random.seed(231)


num_train = 100
num_epoch =30

weight_scale=1e-2
lr = 1e-3
batch_size=20

small_data = {
  'X_train': data['X_train'][:num_train],
  'y_train': data['y_train'][:num_train],
  'X_val': data['X_val'],
  'y_val': data['y_val'],
}

model = ResNet(weight_scale=weight_scale)

solver = Solver(model, small_data,
                num_epochs=num_epoch, batch_size=batch_size,
                update_rule='sgd', #adam
                optim_config={
                  'learning_rate': lr,
                },
                verbose=True, print_every=1)
solver.train()


# %% Plotting the loss, training accuracy, and validation accuracy should show clear overfitting:

plt.subplot(2, 1, 1)
plt.plot(solver.loss_history, 'o')
plt.xlabel('iteration')
plt.ylabel('loss')

plt.subplot(2, 1, 2)
plt.plot(solver.train_acc_history, '-o')
plt.plot(solver.val_acc_history, '-o')
plt.legend(['train', 'val'], loc='upper left')
plt.xlabel('epoch')
plt.ylabel('accuracy')

# save the img
date_today = datetime.datetime.now().strftime("%Y-%m-%d")
figure_name = 'results/'+ date_today + '-' + 'smalloverfit'+ '-ntr' + str(num_train)+ \
    '-e'+ str(num_epoch) + '-ws'+ str(weight_scale) + \
    '-lr'+ str(lr) + '-bs' + str(batch_size) + '.png'
print('Result figure of accuracies and error is saved in: ', figure_name)
plt.savefig(figure_name)

# %% train the net
# ## Train the net
# By training the three-layer convolutional network for one epoch, you should achieve greater than 40% accuracy on the training set:

# In[ ]:


model = ResNet(weight_scale=1e-3)

solver = Solver(model, data,
                num_epochs=1, batch_size=50,
                update_rule='sgd',
                optim_config={
                  'learning_rate': 1e-3,
                },
                verbose=True, print_every=20)
solver.train()

