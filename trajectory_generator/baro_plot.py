"""
this script plots karman mini rocket test flight data that was collected in Feb 2023.
"""
import matplotlib.pyplot as plt
import numpy as np
import csv

t1 = []
pressure= []
# C:\ergasia\projects\Rocket-Trajectory-Tracking-and-Prediction\test_data\kminibaro.txt
with open('C:/ergasia/projects/Rocket-Trajectory-Tracking-and-Prediction/example/kminibaro.txt', 'r') as datafile:
	plotting = csv.reader(datafile, delimiter=',')
	
	for ROWS in plotting:
		t1.append(float(ROWS[0]))
		pressure.append(float(ROWS[1]))
		
# Initialise the subplot function using number of rows and columns
figure, axis = plt.subplots(2,2)

# Pressure plot
axis[0,0].plot(t1, pressure)
axis[0,0].set_title('Atmospheric Pressure',wrap=True)
axis[0,0].set_xlabel('Time(s)')
axis[0,0].set_ylabel('Pressure(Atm)')

# Acceleration plot
t2 = []
accel= []


with open('C:/ergasia/projects/Rocket-Trajectory-Tracking-and-Prediction/example/kminiaccel.txt', 'r') as datafile:
	plotting = csv.reader(datafile, delimiter=',')
	
	for ROWS in plotting:
		t2.append(float(ROWS[0]))
		accel.append(float(ROWS[1]))
		


axis[0,1].plot(t2, accel)
axis[0,1].set_title('Axial Acceleration',wrap=True)
axis[0,1].set_xlabel('Time(s)')
axis[0,1].set_ylabel('Acceleration (m/s^2)')

# # # plot a parabola


# call function calculating the altitude using the pressure data collected 

# Defining the range for the input values on the horizontal axis
x_values = [x for x in range(0, 3000)]
# Computiong the values of the quadratic equation for different values in x_values
y_values = [(-pow(-x,2)+4*x-4) for x in x_values]

axis[1,0].plot(x_values, y_values)
axis[1,0].set_title('Atmospheric Model',wrap=True)
axis[1,0].set_xlabel('Time (s)')
axis[1,0].set_ylabel('Altitude (m)')

# axis[1,1].plot(x_values, x_values)
# axis[1,1].set_title('something',wrap=True)
# axis[1,1].set_xlabel('something')
# axis[1,1].set_ylabel('something')

figure.suptitle('Test Data during Karman Mini 2 flight',weight='bold')
figure.tight_layout()
plt.show()