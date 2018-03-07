PyTorch Notes
=============

# Basic operations
## Basics
```
# Add
result = torch.Tensor(5, 3)
torch.add(x, y, out=result)

# In-place add
y.add_(x)

# torch.view to resize/reshape
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
```

## Numpy Bridge
### Torch Tensor to Numpy Array
__NOTE: Always pay attention to the DTYPE__
```
a = torch.ones(5)
b = a.numpy()
```
### Converting Numpy Array to Torch Tensor
```
a = np.ones(5)
b = torch.from_numpy(a)
```

## CUDA Tensors
Tensors can be moved onto GPU using the `.cuda` method.

# Autograd: automatic differentiation
The `autograd` package provides automatic differentiation for all operations on Tensors. It is a define-by-run framework, which means that your backprop is defined by how your code is run, and that every single iteration can be different.

## Variable
`autograd.Variable` is the central class of the package. It wraps a Tensor, and supports nearly all of operations defined on it. Once you finish your computation you can call `.backward()` and have all the gradients computed automatically.

You can access the raw tensor through the `.data` attribute, while the gradient w.r.t. this variable is accumulated into `.grad`.

If you want to compute the derivatives, you can call `.backward()` on a `Variable`. If `Variable` is a scalar (i.e. it holds a one element data), you don’t need to specify any arguments to `backward()`, however if it has more elements, you need to specify a `gradient` argument that is a tensor of matching shape.

```
# Create a variable
x = Variable(torch.ones(2,2), requires_grad=True)

# Get gradients
gradients = torch.FloatTensor([0.1, 1.0, 0.0001])
y.backward(gradients)
print(x.grad)
```
See more at [here](http://pytorch.org/docs/autograd).

## Define an NN
Please see later nodes in the PDF files.

1. `torch.nn.LSTM`
