# Author: Rambod Azimi
# Date: April 2023
# Learning numpy library in python from 0 to 100

"""
NumPy (Numerical Python) is a Python library for scientific computing that provides support for large, multi-
dimensional arrays and matrices, along with a vast collection of high-level mathematical functions to operate on
these arrays. importing numpy library to our source code so that we can use the features of numpy in our file.
"""

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

