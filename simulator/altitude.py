
from atmo import Atmos

# include modules

import matplotlib.pyplot as plt
import numpy as np
import csv


# plot graphs

# Pressure read
t1 = []
pressure= []
altitude= []

with open("C:/ergasia/projects/Rocket-Trajectory-Tracking-and-Prediction/test_data/kminibaro.txt", 'r') as datafile:
	plotting = csv.reader(datafile, delimiter=',')
	
	for row in plotting:
		t1.append(float(row[0]))
		pressure.append(float(row[1]))
		# calculate altitude at each pressure value
		# altitude.append(Atmos.get_altitude(pressure.index(float(row[1]))))
		atmos = Atmos(float(row[1])) # create atmosphere object with current pressure reading
		altitude.append(atmos.get_altitude())
       		
# Initialise the subplot function using number of rows and columns
figure, axis = plt.subplots(2,2)

# Pressure plot

axis[0,0].plot(t1, pressure)
axis[0,0].set_title('Atmospheric Pressure',wrap=True)
axis[0,0].set_xlabel('Time(s)')
axis[0,0].set_ylabel('Pressure(Atm)')



# Altitude Plot

# create Atmosphere object with pressure values that were read just before
axis[0,1].plot(t1, altitude)
axis[0,1].set_title('Altitude',wrap=True)
axis[0,1].set_xlabel('Time(s)')
axis[0,1].set_ylabel('Altitude WSL (m/s)')


figure.suptitle('Test Data during Karman Mini flight',weight='bold')
figure.tight_layout()
plt.show()






# # plot graphs

# # Pressure read
# t1 = []
# pressure= []

# with open('C:/ergasia/projects/Rocket-Trajectory-Tracking-and-Prediction/test_data/kminibaro.txt', 'r') as datafile:
# 	plotting = csv.reader(datafile, delimiter=',')
	
# 	for ROWS_p in plotting:
# 		t1.append(float(ROWS_p[0]))
# 		pressure.append(float(ROWS_p[1]))
		

# # Initialise the subplot function using number of rows and columns
# figure, axis = plt.subplots(2,2)

# # Pressure plot

# axis[0,0].plot(t1, pressure)
# axis[0,0].set_title('Atmospheric Pressure',wrap=True)
# axis[0,0].set_xlabel('Time(s)')
# axis[0,0].set_ylabel('Pressure(Atm)')


# # Altitude Plot

# # create Atmosphere object with pressure values that were read just before
# atmosphere = Atmos(pressure)

# # calculate altitude at each pressure value
# altitude = [atmosphere.get_altitude() for p in pressure]

# # plot altitude against time
# time = range(len(pressure))
# plt.plot(time, altitude)
# plt.xlabel('Time [s]')
# plt.ylabel('Altitude [m]')
# plt.show()


# figure.suptitle('Test Data during Karman Mini flight',weight='bold')
# figure.tight_layout()
# plt.show()