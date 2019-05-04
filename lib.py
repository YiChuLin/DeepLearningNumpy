"""
This library implements a neural network architecture using numpy.
The architecture is designed so that activation functions, cost functions, and initialization methods can be easily customized
The work currently supports feed forward fully connected network only, future goals include implementing convolutional layers and recurrent functions.
Author: Arvin Lin
"""
import numpy as np

"""
Define Activation Functions
Customized Activation Function could be defined with the following structure:
class customized_activation():
	def __init__(self, ...):
		# init
	def eval(self, z):
		# the function to evaluate the activation
	def grad(self, z):
		# the gradient of the activation
The model will call .eval() when performing forward propagation and call .grad() when performing backward propagation
"""
class binary_step():
	"""
	Usage: m.add('fully_connected', binary_step(step = 1), 20)
	"""
	def __init__(self, step = 1):
		self.step = step
	def eval(self, z):
		return self.step*np.greater_equal(z, 0)
	def grad(self, z):
		return np.diagflat(np.zeros(z.shape))


class linear():
	"""
	Usage: m.add('fully_connected', linear(a = 1), 20)
	"""
	def __init__(self, a = 1):
		self.a = a
	def eval(self, z):	
		return self.a*z
	def grad(self, z):
		return np.diagflat(self.a*np.ones(z.shape))

class sigmoid():
	"""
	Usage: m.add('fully_connected', sigmoid(), 20)
	"""	
	def eval(self, z):
		return 1/(1+np.exp(-z))
	def grad(self, z):
		return np.diagflat(np.exp(-z)/np.square(1+np.exp(-z)))

class tanh():
	"""
	Usage: m.add('fully_connected', tanh(), 20)
	"""	
	def eval(self, z):
		return 2/(1+np.exp(-2*z)) - 1
	def grad(self, z):
		return np.diagflat(4*np.exp(-2*z)/np.square(1+np.exp(-2*z)))

class relu():
	"""
	Usage: m.add('fully_connected', relu(), 20)
	"""	
	def eval(self, z):
		return np.maximum(z, 0)
	def grad(self, z):
		return np.diagflat(np.greater_equal(z, 0))

class leaky_relu():
	"""
	Usage: m.add('fully_connected', leaky_relu(a = 0.001), 20)
	"""	
	def __init__(self, a = 0.001):
		# a is expected to be a positive number less than 1
		self.a = a
	def eval(self, z):
		return np.maximum(z, self.a*z)
	def grad(self, z):
		return np.diagflat(np.greater_equal(z, 0) + self.a*np.less(z, 0))

class softmax():
	"""
	Usage: m.add('fully_connected', softmax(stable = True), 20)
	"""		
	def __init__(self, stable = True):
		self.stable = stable
	def eval(self, z):
		if self.stable == True:
			z = z - np.max(z)
			return np.exp(z)/sum(np.exp(z))
	def grad(self, z):
		s = self.eval(z).reshape(-1, 1)
		return np.diagflat(s) - np.dot(s, s.T)

# The following are some sample customized activation function that are solely for demonstration purposes.
# These activation functions' performance are not necessary satisfiable
class resu():
	"""
	# Rectified Square Unit
	Usage: m.add('fully_connected', resu(), 20)
	"""	
	@staticmethod
	def eval(z):
		return np.square(np.maximum(z,0))
	@staticmethod
	def grad(z):
		return np.diagflat(2*np.maximum(z,0))

class repu():
	"""
	# Rectified Polynomial Unit
	Usage: m.add('fully_connected', repu(a = 2), 20)
	"""		
	def __init__(self, a = 2):
		self.a = a	
	def eval(self, z):
		return np.power(np.maximum(z,0), self.a)
	def grad(self, z):
		return np.diagflat(self.a*np.power(np.maximum(z,0), self.a-1))
"""
Define Cost Functions
Customized Cost Function should have the following structure
class cost_func():
	def eval(self, a, e):
		#evaluate the cost
	def grad(self, a, e):
		#evaluate the gradient of the cost function
"""
class quadratic_cost():
	"""
	Usage: m.initialize_cost(cost_function = quadratic_cost())
	"""
	def eval(self, a, e):
		return 0.5*np.sum(np.square(a-e))
	def grad(self, a, e):
		return a - e
class cross_entropy_cost():
	"""
	Usage: m.initialize_cost(cost_function = cross_entropy_cost())
	"""
	def eval(self, a, e):
		a = np.clip(a, 1e-15, 1-1e-15)
		return -np.sum(np.log(a)*e + (1-e)*np.log(1-a))
	def grad(self, a, e):
		with np.errstate(divide='ignore', invalid = 'ignore'):
			c = np.true_divide(a-e, (1-a)*a)
			c = np.nan_to_num(c)
			return c
"""
Initialization Methods
"""
def He_initialization(dim):
	"""
	Usage: m.initialize_weights(He_initialization)
	"""
	prev_dim = dim[-1]
	dim = np.prod(dim)
	return np.random.randn(dim)*np.sqrt(2/prev_dim)
def random_initialization(dim):
	"""
	Usage: m.initialize_weights(random_initialization)
	"""
	dim = np.prod(dim)
	return np.random.randn(dim)

"""
Define Model
"""
class model():
	"""
	Sample Usage:
	m = model()
	m.add('input', node_number = x_size)
	m.add('fully_connected', relu(), 20)
	m.add('fully_connected', relu(), 20)
	m.add('fully_connected', softmax(), Y.shape[1])
	m.initialize_weights(He_initialization)
	
	m.initialize_cost(cost_function = cross_entropy_cost())
	m.gradient_check(X[0], Y[0])

	m.Train(X, Y, lr = 0.01, epoch = 50)
	m.evaluate(Xt, Yt) 
	"""
	def __init__(self):
		self.layer_count = 0
		self.layers_spec = []
		self.layers = []
		self.weights = np.array([])
		self.cost_function = None
	def add(self, layer_type = 'fully_connected', activation = relu, node_number = 10):
		if layer_type == 'input':
			layer_spec = {'type':'input', 'name':'input_'+str(self.layer_count), 'node_number': node_number}
			self.layers_spec.append(layer_spec)
		elif layer_type == 'fully_connected':
			layer_spec = {'type':'fully_connected','name':'fully_connected_'+str(self.layer_count), 'activation': 										activation, 'node_number':node_number}
			self.layers_spec.append(layer_spec)
		else:
			print('Layer Type: '+str(layer_type)+' does not exist')
		self.layer_count += 1
	def initialize_weights(self, method = random_initialization):
		self.layers = []
		self.weights = np.array([])
		try:
			generator = method

		except:
			generator = random_initialization
			print('Initialization method unknown, Using random_initialization instead!')	
		previous_node_num = None		
		for layer_spec in self.layers_spec:
			if previous_node_num == None:
				previous_node_num = layer_spec['node_number']
			else:
				if layer_spec['type'] =='fully_connected':
					layer = {'weight_size':[layer_spec['node_number'],previous_node_num],
						'bias_size':layer_spec['node_number'],
						'activation':layer_spec['activation']}
					weight = generator((layer_spec['node_number'],previous_node_num))
					bias = np.zeros(layer_spec['node_number'])
					self.layers.append(layer)
					self.weights = np.concatenate((self.weights, weight, bias))
					previous_node_num = layer_spec['node_number']

	def initialize_cost(self, cost_function = quadratic_cost()):
		self.cost_function = cost_function
	def feed_forward(self, layer, inputs, caches, weight_count):
		#Unpack values
		activate = layer['activation']
		b_size = layer['bias_size']
		w_size = layer['weight_size']
		weight = self.weights[weight_count: weight_count + w_size[0]*w_size[1]].reshape(w_size)
		weight_count += w_size[0]*w_size[1]
		bias = self.weights[weight_count: weight_count + b_size].reshape([b_size, 1])
		z = np.dot(weight, inputs)+bias
		weight_count += b_size
		cache = {'x':inputs, 'wx':z, 'activation':activate, 'w':weight}
		caches.append(cache)
		return activate.eval(z = z), caches, weight_count
	def run(self, inputs):
		if self.layers == []:
			print('Please initialize the layers before running')
		else:
			curr_value = inputs.reshape(inputs.shape[0],1)
			caches = []
			weight_count = 0
			for layer in self.layers:
				curr_value, caches, weight_count = self.feed_forward(layer = layer, inputs = curr_value, caches = caches, weight_count = weight_count)
		return curr_value, caches

	def feed_backward(self, delta_next, cache):
		x = cache['x']
		wx = cache['wx']
		w = cache['w']
		activate = cache['activation']
		if w == []:
			delta = np.dot(activate.grad(wx), delta_next)			
			#delta = np.multiply(activate.grad(wx), delta_next)
		else:
			delta = np.dot(activate.grad(wx), np.matmul(w.T, delta_next))
			#delta = np.multiply(np.matmul(weights.T, delta_next),activate.grad(wx))
		grad_w = np.matmul(delta, x.T)
		#self.weights[i]['weight'] -= grad_w*lr
		#self.weights[i]['bias']   -= delta*lr
		#print(grad_w)
		return delta, grad_w
	def backward_prop(self, a, e, caches):
		delta = self.cost_function.grad(a, e)
		i = len(self.layers)
		grads = []
		w = []
		for cache in reversed(caches):
			next_w = cache['w']
			cache['w'] = w
			delta, grad_w = self.feed_backward(delta_next = delta, cache = cache)
			w = next_w
			grads = np.concatenate((grad_w.flatten(), delta.flatten(), grads))
		return grads

	def gradient_check(self, X, Y, epsilon=1e-7):
		Y = Y.reshape(Y.size, 1)
		curr_value, caches = self.run(inputs = X)
		grads = self.backward_prop(curr_value, Y, caches)
		grads_approx = np.array([])
		for i in range(len(self.weights)):
			temp = self.weights[i]
			self.weights[i] = temp + epsilon
			z_plus,_ = self.run(X)
			J_plus = self.cost_function.eval(z_plus, Y)
			self.weights[i] = temp - epsilon
			z_minus,_ = self.run(X)
			J_minus = self.cost_function.eval(z_minus, Y)
			self.weights[i] = temp
			grad_approx = np.array([(J_plus - J_minus)/(2*epsilon)])
			grads_approx = np.concatenate((grads_approx, grad_approx))
		numerator = np.linalg.norm(grads - grads_approx) 
		denominator = np.linalg.norm(grads) + np.linalg.norm(grads_approx)
		difference = numerator/denominator
		print('Difference between gradient and gradient approximation: ' + str(difference))
	def Train(self, X, Y, lr = 0.0001, epoch = 10, print_cost = True):
		for epo in range(epoch):
			accumulated_loss = 0
			accurate_num = 0
			for x, y in zip(X, Y):
				curr_value, caches = self.run(inputs = x)
				y = y.reshape(y.size, 1)
				grads = self.backward_prop(a = curr_value, e = y, caches = caches)
				self.weights -= grads*lr
				if print_cost == True:
					accumulated_loss += self.cost_function.eval(curr_value, y)
					curr_value = (curr_value == max(curr_value))
					if np.array_equal(curr_value, y):
						accurate_num += 1
			print('Current Loss: '+str(accumulated_loss/X.shape[0])+'	accuracy: '+str(float(accurate_num)/X.shape[0]))
	def evaluate(self, Xt, Yt):
		accurate_num = 0
		accumulated_cost = 0
		for x, y in zip(Xt, Yt):
			y_pred, _ = self.run(inputs = x)
			y = y.reshape(y.size, 1)
			accumulated_cost += self.cost_function.eval(y_pred, y)
			y_pred = (y_pred == max(y_pred))
			if np.array_equal(y_pred, y):
				accurate_num += 1
		print('Total Loss: '+str(accumulated_cost/Xt.shape[0])+'	accuracy: '+str(float(accurate_num)/Xt.shape[0]))

