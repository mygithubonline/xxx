'''
import numpy as np
import matplotlib.pyplot as plt
x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
plt.plot(x0)
plt.show()
'''

'''
string1 = "Linux"
string2 = " Hint"
joined_string = string1 + string2
print(joined_string)
'''

'''
# Use of String Formatting
float1 = 563.78453
print("{:5.2f}".format(float1))
# Use of String Interpolation
float2 = 563.78453
print("%5.2f" % float2)
'''

import math
# Assign values to x and n
x = 4
n = 3

# Method 1
power = x ** n
print("%d to the power %d is %d" % (x,n,power))

# Method 2
power = pow(x,n)
print("%d to the power %d is %d" % (x,n,power))

# Method 3
power = math.pow(2,6.5)
print("%d to the power %d is %5.2f" % (x,n,power))