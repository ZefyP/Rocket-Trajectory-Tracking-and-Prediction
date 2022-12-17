# this program implements a Kalman filter for predicting the altitude of a rocket using pressure, 
# 3 axis acceleration, time, and temperature data:

# Import the necessary libraries
import numpy as np
from numpy.linalg import inv

# Kalman filter class
class KalmanFilter:
  def __init__(self, state_transition, observation, initial_state_mean, initial_state_covariance, process_noise, measurement_noise):
    # Store the state transition, observation, and noise matrices
    self.state_transition = state_transition
    self.observation = observation
    self.process_noise = process_noise
    self.measurement_noise = measurement_noise
    
    # Initialize the state mean and covariance
    self.state_mean = initial_state_mean
    self.state_covariance = initial_state_covariance
    
  def predict(self, control_input):
    # Use the state transition matrix to predict the next state
    # In this code, the @ operator is being used to multiply the state transition matrix by the 
    # state mean, then adding the control input. This is used to predict the next state.
    self.state_mean = self.state_transition @ self.state_mean + control_input
    
    # Use the process noise to compute the state covariance
    self.state_covariance = self.state_transition @ self.state_covariance @ self.state_transition.T + self.process_noise
  
  def update(self, measurement):
    # Compute the Kalman gain
    kalman_gain = self.state_covariance @ self.observation.T @ inv(self.observation @ self.state_covariance @ self.observation.T + self.measurement_noise)
    
    # Update the state mean and covariance based on the measurement
    self.state_mean = self.state_mean + kalman_gain @ (measurement - self.observation @ self.state_mean)
    self.state_covariance = self.state_covariance - kalman_gain @ self.observation @ self.state_covariance
