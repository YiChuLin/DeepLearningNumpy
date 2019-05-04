# DeepLearningNumpy
Implements deep learning models in pure numpy

## Currently Supports
### Neural Network Types
Fully Connected Layers

### Activations
- binary step  
- linear  
- sigmoid  
- tanh  
- relu  
- leaky_relu  
- softmax  
Self defined activation functions could be created with the following structure:
```
class activation():
  def __init__(self, ...)
    ...
  def eval(self, z):
    # evaluate on input z
    return ...
  def grad(self, z):
    # calculate the gradient of the activation given z, should return a diagnal matrix given a vector
    return ...
```
### Cost Functions
- quadratic cost
- cross entropy cost
Self defined cost functions should follow the structure similarly to activation functions

### Initializations
- He_initialization
- random_initialization

## Sample Usage
```
# Define the fully connected network architecture

m = model()
m.add('input', node_number = x_size)
m.add('fully_connected', relu(), 20)
m.add('fully_connected', relu(), 20)
m.add('fully_connected', softmax(), Y.shape[1])
m.initialize_weights(He_initialization)

#Initialize Cost
m.initialize_cost(cost_function = cross_entropy_cost())

#Check if the model performs back propagation correctly
m.gradient_check(X[0], Y[0])

m.Train(X, Y, lr = 0.01, epoch = 50)
m.evaluate(Xt, Yt) 

```
