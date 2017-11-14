# consider neural network as graphs where nodes are operations
# and data (referred to as tensor, represented in the form of multi-dimensional arrays) flow along the edges
# importing tensorflow and inititalizing two variables which are constants
# here some basics operations like multiplications and addition using tensorflow

import tensorflow as tf

# initialise constants
x1 = tf.constant([1,2,3,4])     # data or tensor in the form of array
x2 = tf.constant([5,6,7,8])

# Multiplication operation
x = tf.multiply(x1,x2)

print(x,"Initialise session, print 'x' and close the session")

# The result of the lines of code is an abstract tensor in the computation graph.
# when you execute the code the result (x) doesnt actually get calculated.
# Tf just defined the model but no process has run to calculate x1.

# How to calculate and see the result?

# 1) initialize the session
session = tf.Session()

# 2) PRINT x in this session

print(session.run(x), 'Calculated in session')

# 3) Close the session

# OR Startup a session, run x and close the session automatically after printing the output

with tf.Session() as session :
    output = session.run(x)
    print(output, "Initialise session and run 'x'")



#-------------------- Custom Configuration for Session -----------------------------------------------------

# till now we inititalise default session, but we can specify the configuration of a particular session
# eg.

config =  tf.ConfigProto(log_device_placement = True)
print(config)
# this configuration helps to LOG GPU or CPU device that is aasigned to the operation.
# helps in knowing information about which devices are used in the session for each operation

# FOR SOFT CONSTRAINTS FOR DEVICE PLACEMENT USE
config = tf.ConfigProto(allow_soft_placement = True)

# NEXT EXPLORE AND UNDERSTANDING DATA    