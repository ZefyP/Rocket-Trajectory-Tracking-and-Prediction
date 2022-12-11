# Import the necessary libraries
import numpy as np

# In the context of a Kalman filter, "process noise" refers to the uncertainty in the system dynamics that is not accounted for 
# by the state transition matrix. It represents the unpredictable variations in the system over time, such as random disturbances 
# or unmodeled effects. "Measurement noise" refers to the uncertainty in the measurements that are used to update the state estimate. 
# It represents the error or inconsistency in the observations, such as sensor noise or bias.

# The values of the process and measurement noise matrices depend on the specific system being modeled and the accuracy of the model 
# and measurements. In general, the process noise should be set based on the expected variability of the system dynamics, and the 
# measurement noise should be set based on the expected accuracy of the measurements.

# Here is an example of how the process and measurement noise matrices could be defined for a rocket that is being tracked using 
# pressure, 3 axis acceleration, time, and temperature data ( = 5 state variables):




# Define the process noise matrix
process_noise = np.array([[0.1, 0.0, 0.0, 0.0, 0.0],
                           [0.0, 0.1, 0.0, 0.0, 0.0],
                           [0.0, 0.0, 0.1, 0.0, 0.0],
                           [0.0, 0.0, 0.0, 0.1, 0.0],
                           [0.0, 0.0, 0.0, 0.0, 0.1]])

# Define the measurement noise matrix
measurement_noise = np.array([[0.01, 0.0, 0.0, 0.0, 0.0],
                              [0.0, 0.01, 0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.01, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 0.01, 0.0],
                              [0.0, 0.0, 0.0, 0.0, 0.01]])


# These values are just examples and may not be appropriate for our specific situation. 
# We may need to adjust the values based on the characteristics of our system and the quality of our data.



# This is an attempt to calculate the process and measurement noise matrices:


# Define the expected variability of the system dynamics
# Can be estimated by computing the standard deviation or other statistical measures of the observed variations. 
# This estimated value can then be used as the process noise in the kalman filter.
dynamic_variability = 0.1

# Define the expected accuracy of the measurements
measurement_accuracy = 0.01

# Create a 5x5 matrix of zeros for the process noise matrix
process_noise = np.zeros((5, 5))

# Set the diagonal elements of the process noise matrix to the expected variability of the system dynamics
process_noise[0, 0] = dynamic_variability
process_noise[1, 1] = dynamic_variability
process_noise[2, 2] = dynamic_variability
process_noise[3, 3] = dynamic_variability
process_noise[4, 4] = dynamic_variability

# Create a 5x5 matrix of zeros for the measurement noise matrix
measurement_noise = np.zeros((5, 5))

# Set the diagonal elements of the measurement noise matrix to the expected accuracy of the measurements
measurement_noise[0, 0] = measurement_accuracy
measurement_noise[1, 1] = measurement_accuracy
