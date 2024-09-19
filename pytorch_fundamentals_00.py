import torch

print(torch.__version__)
#Scalar is a single number
scalar = torch.tensor(7)
print(scalar)
#We can check its dimensions using
print(scalar.ndim)

#Para obtener el numero del tensor (solamente funciona con tensores de un elemento
print(scalar.item())

#Vector
vector = torch.tensor([7, 7])
print(vector)

#Check shape of vector it tells you how elements inside are arranged
#Es como el length
print(vector.shape)

MATRIX = torch.tensor([[7,8],[9, 10]])
print(MATRIX.shape)

#TENSOR
TENSOR = torch.tensor([[1, 2, 3],
                       [3, 6, 9],
                       [2, 4, 5]])
print(TENSOR)

#Create a random tensor of size  (3,4)
random_tensor = torch.rand(size=(3,4))
print(random_tensor)
print(random_tensor.dtype)

#Create a tensor of all zeros
zeros = torch.zeros(size=(3,4))
print(zeros)
print(zeros.dtype)

ones = torch.ones(size=(3,4))
print(ones)
print(ones.dtype)

#Create a range of values
zero_to_ten = torch.arange(start=0, end=10, step=2)
print(zero_to_ten)

#Can also create a tensor of zeros similar to another tensor
# Can also create a tensor of zeros similar to another tensor
ten_zeros = torch.zeros_like(input=zero_to_ten) # will have same shape
print(ten_zeros)

#Create a tensor
some_tensor = torch.rand(3,4)

#Find details about it
print(some_tensor)
print(f"Shape of tensor: {some_tensor.shape}")
print(f"Datatype of tensor: {some_tensor.dtype}")
print(f"Device tensor is stored on: {some_tensor.device}")

#Basic operations with tensors
print("Element-wise multiplication * or torch.mul")
tensor = torch.tensor([1, 2, 3])
print(tensor, "*", tensor)
print("Equals:", tensor * tensor)

print("Matrix multiplication operator @ or torch.matmul() ")
tensor = torch.tensor([1, 2, 3])
print(torch.matmul(tensor, tensor))

print("One of the most common errors in DL are shape errors")

# Shapes need to be in the right way
tensor_A = torch.tensor([[1, 2],
                         [3, 4],
                         [5, 6]], dtype=torch.float32)

tensor_B = torch.tensor([[7, 10],
                         [8, 11],
                         [9, 12]], dtype=torch.float32)
# View tensor_A and tensor_B
print(tensor_A)
print(tensor_B)

print(tensor_A)
print(tensor_B.T)

# The operation works when tensor_B is transposed
print(f"Original shapes: tensor_A = {tensor_A.shape}, tensor_B = {tensor_B.shape}\n")
print(f"New shapes: tensor_A = {tensor_A.shape} (same as above), tensor_B.T = {tensor_B.T.shape}\n")
print(f"Multiplying: {tensor_A.shape} * {tensor_B.T.shape} <- inner dimensions match\n")
print("Output:\n")
output = torch.matmul(tensor_A, tensor_B.T)
print(output)
print(f"\nOutput shape: {output.shape}")
print("---------------------------------")
# Since the linear layer starts with a random weights matrix, let's make it reproducible (more on this later)
torch.manual_seed(42)
# This uses matrix multiplication
linear = torch.nn.Linear(in_features=2, # in_features = matches inner dimension of input
                         out_features=6) # out_features = describes outer value
x = tensor_A
output = linear(x)
print(f"Input shape: {x.shape}\n")
print(f"Output:\n{output}\n\nOutput shape: {output.shape}")
