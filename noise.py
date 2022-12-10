# In the context of a Kalman filter, "process noise" refers to the uncertainty in the system dynamics that is not accounted for 
# by the state transition matrix. It represents the unpredictable variations in the system over time, such as random disturbances 
# or unmodeled effects. "Measurement noise" refers to the uncertainty in the measurements that are used to update the state estimate. 
# It represents the error or inconsistency in the observations, such as sensor noise or bias.

# The values of the process and measurement noise matrices depend on the specific system being modeled and the accuracy of the model 
# and measurements. In general, the process noise should be set based on the expected variability of the system dynamics, and the 
# measurement noise should be set based on the expected accuracy of the measurements.

# Here is an example of how the process and measurement noise matrices could be defined for a rocket that is being tracked using 
# pressure, 3 axis acceleration, time, and temperature data:

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