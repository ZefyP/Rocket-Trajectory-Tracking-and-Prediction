# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

'''
This function takes in the pressure measurements as well as the process noise.
Args:
    Q: covariance matrix 
    R: the measurement noise covariance matrix
    The function returns the 
estimated altitude and error at each time step.
'''
def kalman_filter(pressure, Q, R):
    # Define initial state and covariance matrix
    x = np.array([0, 0])    # [altitude, velocity]
    P = np.diag([100, 10])  # Initial covariance matrix
    I = np.eye(2)           # Identity matrix
    
    # Define measurement matrix
    H = np.array([[1, 0]])
    
    # Initialize output variables
    altitude_estimate = [x[0]]
    error_estimate = [P[0,0]]
    
    # Loop through pressure measurements and update state estimate
    for i in range(len(pressure)):
        # Prediction step
        x = np.dot(x, np.array([[1, 1], [0, 1]]))
        P = np.dot(np.dot(np.array([[1, 1], [0, 1]]), P), np.array([[1, 1], [0, 1]]).T) + Q
        
        # Measurement step
        K = np.dot(np.dot(P, H.T), np.linalg.inv(np.dot(np.dot(H, P), H.T) + R))
        x = x + np.dot(K, (pressure[i] - x[0]))
        P = np.dot((I - np.dot(K, H)), P)
        
        # Save altitude estimate and error estimate for plotting
        altitude_estimate.append(x[0])
        error_estimate.append(P[0,0])
    
    return altitude_estimate, error_estimate


'''
This function takes in the measured altitude and the predicted altitude, 
and returns the absolute difference between the two.
'''
def calculate_error(measured_altitude, predicted_altitude):
    error = np.abs(measured_altitude - predicted_altitude)
    return error


'''
takes in the measured altitude and the predicted altitude, 
calculates the error using the calculate_error function, 
and plots the error over time.
'''
def plot_error(measured_altitude, predicted_altitude):
    error = calculate_error(measured_altitude, predicted_altitude)
    plt.plot(error)
    plt.xlabel('Time (s)')
    plt.ylabel('Error (m)')
    plt.title('Kalman Filter Error')
    plt.show()


# Generate fake data arrays and call the fucctions

# Generate fake data arrays
pressure = np.array([1000, 950, 900, 850, 800, 750, 700, 650, 600, 550])
Q = np.diag([1, 1])
R = np.array([[10]])
measured_altitude = np.array([0, 100, 200, 300, 400, 500, 600, 700, 800, 900])
predicted_altitude = np.array([50, 150, 250, 350, 450, 550, 650, 750, 850, 950])

# Call Kalman filter function with fake data
altitude_estimate, error_estimate = kalman_filter(pressure, Q, R)

# Calculate error between measured and predicted altitudes
error = np.abs(measured_altitude - predicted_altitude)

# Plot error over time
plt.plot(error, label='Measurement Error')
plt.plot(error_estimate, label='Kalman Filter Error')
plt.xlabel('Time (s)')
plt.ylabel('Error (m)')
plt.title('Kalman Filter Error')
plt.legend()
plt.show()