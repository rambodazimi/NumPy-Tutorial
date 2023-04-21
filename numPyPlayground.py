# Author: Rambod Azimi
# Date: April 2023
# Learning numpy library in python from 0 to 100

"""
NumPy (Numerical Python) is a Python library for scientific computing that provides support for large, multi-
dimensional arrays and matrices, along with a vast collection of high-level mathematical functions to operate on
these arrays. importing numpy library to our source code so that we can use the features of numpy in our file.
NumPy works with different matrices which makes it so powerful to do scientific operations on matrices. It relies on 
ndarray (N-dimensional arrays). We can plot figures, do some backends, and write Machine Learning algorithms all with Numpy.
"""

# if you have not already installed numpy on your computer, open terminal (command line) and type "pip install numpy"

# importing the numpy library to our source code so that we can use the methods in it in our program
import numpy as np # name it as "np" for simplicity

print("Welcome to NumPy tutorial!")
print("Author: Rambod Azimi")
print("Date: April 2023\n")

# Creating lists of integers named int_list and int_list2
int_list = [1, 2, 3, 4, 5]
int_list2 = [6, 7, 8, 9, 10]

# printing both lists
print("First list:", int_list)
print("Second list:", int_list2)

# doing some simple operation using built-in methods in numpy library
print("Max value of the first list:", np.amax(int_list))
print("Min value of the first list:", np.amin(int_list))
print("Mean value of the first list:", np.mean(int_list))
print("Median value of the first list:", np.median(int_list))
print("Standard deviation of the first list:", np.std(int_list))
print("Variance of the first list:", np.var(int_list))
print("Absolute value of -5 is:", np.abs(-5))
print("Adding the first list with 10:", np.add(int_list, 10))
print("Appending 6 to the end of the first list:", np.append(int_list, 6))
print("Comparing 2 lists with each other and retun a boolean:", np.array_equiv(int_list, int_list2)) # should be false
print("Comparing 2 lists with each other and retun a boolean:", np.array_equiv(int_list, [1, 2, 3, 4, 5])) # should be true
print("Summing the elements of the first list:", np.sum(int_list)) # should be 15
print("Summing the elements of the second list:", np.sum(int_list2)) # should be 40
print("Dot product of 2 lists:", np.dot(int_list, int_list2)) # both lists should have the save size
print("Size of the first list:", np.size(int_list))
print("Size of the second list:", np.size(int_list2))
print("Sorting the list [5, 2, 3, 1] ->", np.sort([5, 2, 3, 1]))
print()
# There are many more built in functions which we'll see during this tutorial...

# print(int_list * int_list2)
# This will result in error because we can't multiply 2 lists together

# creating list using Numpy library with array() method. my_array and my_array2 are somehow matrices...
my_array = np.array([5, 4, 3, 2, 1])
my_array2 = np.array([1, 2, 3, 4, 5])

# printing the arrays
print("First array created using Numpy array():", my_array)
print("Second array created using Numpy array():", my_array2)

# Multiplying 2 arrays together (not possible in a regular list)
print("Multiplying 2 arrays together:", my_array * my_array2)

# creating an array of arrays (2d array)
array2d = np.array([[1.5, 2.5, 3.5], [5.5, 6.5, 7.5]])
print("2D array:\n", array2d)
print("Dimension of the array is:", array2d.ndim)
print("Number of rows and columns are:", array2d.shape)
print("Type of the 2d-array is:", array2d.dtype) # float64 means double precision floating point number (64 bits or 8 bytes)
print("Size of the 2d array is:", array2d.size) # number of elements

# getting a specific elemtent of a 2d array
print("array2d[1,2] is:", array2d[1,2]) # index starts from 0
print("First row of the 2d array is:", array2d[0,:])
print("First column of the 2d array is:", array2d[:,0])
array2d[0,0] = 9.5 # changing the first element of the 2d array to 9.5
print("Array with the changed first element is:\n", array2d)
print()

# Now, working a little bit on matrix
print("Creating a matrix of zeros:", np.zeros(5))
print("Creating a 2d matrix of zeros:\n", np.zeros((2,5))) # 2 rows and 5 columns
print("Creating a 2d matrix of ones:\n", np.ones((2, 4)))
print("Creating a 2d matrix of a specific values:\n", np.full((2, 3), 20)) # creating a 2x3 matrix with values of 20

# Creating a matrix of random numbers between 0 and 1
print(np.random.rand(3, 3))

# Creating a random integer number between 0 and 10
print("A random int in [0, 10]:", np.random.randint(0, 10))
# Creating a 3x3 matrix of random integer numbers between 0 and 10
print("A random 3x3 integer matrix in [0, 10]:\n", np.random.randint(0, 10, size=(3, 3)))
print()

# Exercise 1:
"""
use numpy library in python to create a matrix like that:
matrixA = [1 1 1 1 1
           1 0 0 0 1
           1 0 9 0 1
           1 0 0 0 1
           1 1 1 1 1]
"""
print("Exercise:")
matrixA = np.zeros((5, 5))
matrixA[0,:] = 1 # making the first row to be 1
matrixA[matrixA.shape[1]-1, :] = 1 # making the last row to be 1 (shape[1] returns the number of columns). -1 because index starts from 0
matrixA[:, 0] = 1 # making the first column to be 1
matrixA[:, matrixA.shape[0]-1] = 1 # making the last column to be 1

print(matrixA)
print()

# Now, let's work a bit about shallow and deep copy

# shallow copy
arr = np.array([1, 2, 3]) # creating a simple array
shallow_copy = arr # shallow copy
shallow_copy[0] = 100 # changing the first elements of the shallow_copy array
print("shallow_copy:", shallow_copy) 
print("Original array:", arr) # arr has also changed!!

# deep copy
arr = np.array([1, 2, 3]) # creating a simple array
deep_copy = arr.copy()
deep_copy[0] = 100
print("deep_copy:", deep_copy)
print("Original array:", arr) # arr has not changed
print()

# Now let's talk a bit about linear Algebra using linalg

# Matrix multiplication
matrix1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
matrix2 = np.array([[-1, 2, 0], [4, 0, 1], [1, -2, 0]])

print("First matrix:\n", matrix1)
print("Second matrix:\n", matrix2)
print("Matrix multiplication:\n", np.matmul(matrix1, matrix2)) # note that sizes should be compatible!
print("Matrix transpose:", matrix1.T)
print("Condition number of the first matrix:", np.linalg.cond(matrix1))
print("Determinent of the first matrix:", np.linalg.det(matrix1)) # Almost 0 because is invertible
print("Eigenvalues of the first matrix:", np.linalg.eigvals(matrix1))
print("Inverse of [1 2 ; 3 4]:\n", np.linalg.inv([[1, 2], [3, 4]]))
print("Identity matrix of size 3:\n", np.identity(3))
print()

# Finally, let's talk about some statistics
