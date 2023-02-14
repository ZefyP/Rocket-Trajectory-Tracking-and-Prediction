import tkinter as tk
from tkinter import ttk
import simulator

# Create the main window
window = tk.Tk()
window.title("Rocket Trajectory Tracking and Prediction")
window.geometry("800x600")

# Set the font to Montserrat
font = ("Montserrat", 14)

# Create the main frame
frame = ttk.Frame(window, padding=20)
frame.grid()

# Add a label to display the current status of the rocket
status_label = ttk.Label(frame, text="Status: Launched", font=font)
status_label.grid(row=0, column=0)


# Create the predict button
predict_button = ttk.Button(frame, text="Predict Trajectory", font=font, command=simulator.predict_trajectory)
predict_button.grid(row=1, column=0)


# Add a label to display the predicted trajectory
trajectory_label = ttk.Label(frame, text="Predicted Trajectory: ", font=font)
trajectory_label.grid(row=2, column=0)

# Add a canvas to display the rocket's current position on a map
map_canvas = tk.Canvas(frame, width=500, height=500)
map_canvas.grid(row=0, column=1, rowspan=3)

# Run the main loop
window.mainloop()







# --------------------


# import tkinter as tk
# from tkinter import ttk

# # Create the main window
# window = tk.Tk()
# window.title("Rocket Trajectory Tracking and Prediction")
# window.geometry("800x600")

# # Set the font to Montserrat
# font = ("Montserrat", 14)

# # Create the main frame
# frame = ttk.Frame(window, padding=20)
# frame.grid()

# # Add a label to display the current status of the rocket
# status_label = ttk.Label(frame, text="Status: Launched")
# status_label.configure(font=font)
# status_label.grid(row=0, column=0)

# # Add a button to predict the future trajectory of the rocket
# predict_button = ttk.Button(frame, text="Predict Trajectory")
# predict_button.configure(font=font)
# predict_button.grid(row=1, column=0)

# # Add a label to display the predicted trajectory
# trajectory_label = ttk.Label(frame, text="Predicted Trajectory: ")
# trajectory_label.configure(font=font)
# trajectory_label.grid(row=2, column=0)

# # Add a canvas to display the rocket's current position on a map
# map_canvas = tk.Canvas(frame, width=500, height=500)
# map_canvas.grid(row=0, column=1, rowspan=3)

# # Run the main loop
# window.mainloop()



# ----------------------------
# # Import the necessary libraries
# import tkinter as tk
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# class RocketGui:
#   def __init__(self, master):
#     # Create the main window
#     self.master = master
#     self.master.geometry('600x600')
#     self.master.title('Rocket GUI')

#     # Create the duration label
#     self.duration_label = tk.Label(self.master, text='Duration: 0 s')
#     self.duration_label.pack()

#     # Create the map frame
#     self.map_frame = tk.Frame(self.master, width=400, height=400)
#     self.map_frame.pack()

#     # Create the map figure and canvas
#     self.map_figure = Figure(figsize=(4, 4))
#     self.map_canvas = FigureCanvasTkAgg(self.map_figure, self.map_frame)
#     self.map_canvas.draw()
#     self.map_canvas.get_tk_widget().pack()

#     # Create the status window
#     self.status_window = tk.Text(self.master)
#     self.status_window.pack()

#     # Create the flight log
#     self.flight_log = []

#   def draw_trajectory(self, trajectory):
#     # Clear the map figure
#     self.map_figure.clf()

#     # Draw the trajectory on the map figure
#     ax = self.map_figure.add_subplot(111)
#     ax.plot(trajectory[:, 0], trajectory[:, 1])

#     # Redraw the map canvas
#     self.map_canvas.draw()

#   def update_status(self, status):
#     # Append the status to the flight log
#     self.flight_log.append(status)

#     # Clear the status window
#     self.status_window.delete(1.0, tk.END)

#     # Update the status window with the flight log
#     for log in self.flight_log:
#       self.status_window.insert(tk.END, log)

#   def update_duration(self, duration):
#     # Update the duration label
#     self.duration_label.configure(text='Duration: {} s'.format(duration))

# ---------------------
# import sys
# from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QLineEdit, QPushButton
# from PyQt5.QtGui import QFont

# from rocket_trajectory_tracking_and_prediction.trajectory_prediction import predict

# class MainWindow(QMainWindow):
#     def __init__(self):
#         super().__init__()
#         self.initUI()

#     def initUI(self):
#         # Create widgets
#         self.label = QLabel("Enter mission parameters:", self)
#         self.line_edit = QLineEdit(self)
#         self.button = QPushButton("Predict Trajectory", self)

#         # Set widget properties
#         self.label.setFont(QFont("Arial", 20))
#         self.line_edit.setFont(QFont("Arial", 20))
#         self.button.setFont(QFont("Arial", 20))
#         self.line_edit.setMinimumWidth(500)
#         self.button.setMinimumWidth(500)

#         # Set widget positions
#         self.label.move(20, 20)
#         self.line_edit.move(20, 70)
#         self.button.move(20, 120)

#         # Connect button click to function
#         self.button.clicked.connect(self.predictTrajectory)

#         # Set window properties
#         self.setGeometry(300, 300, 550, 200)
#         self.setWindowTitle("Trajectory Prediction Tool")
#         self.show()

#     def predictTrajectory(self):
#         # Get mission parameters from line edit
#         mission_parameters = self.line_edit.text()

#         # Perform trajectory prediction using mission parameters
#         predicted_trajectory = predict(mission_parameters)

#         # Display result in window
#         self.label.setText("Predicted Trajectory: " + predicted_trajectory)

# if __name__ == '__main__':
#     app = QApplication(sys.argv)
#     window = MainWindow()
#     sys.exit(app.exec_())


# -------------------------------
# # Import the necessary libraries
# import tkinter as tk
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# from matplotlib.figure import Figure

# class RocketGui:
#   def __init__(self, master):
#     # Create the main window
#     self.master = master
#     self.master.geometry('600x600')
#     self.master.title('Rocket GUI')

#     # Create the duration label
#     self.duration_label = tk.Label(self.master, text='Duration: 0 s')
#     self.duration_label.pack()

#     # Create the map frame
#     self.map_frame = tk.Frame(self.master, width=400, height=400)
#     self.map_frame.pack()

#     # Create the map figure and canvas
#     self.map_figure = Figure(figsize=(4, 4), dpi=100)
#     self.map_canvas = FigureCanvasTkAgg(self.map_figure, self.map_frame)
#     self.map_canvas.draw()
#     self.map_canvas.get_tk_widget().pack()

#     # Create the status window
#     self.status_window = tk.Text(self.master, width=40, height=10)
#     self.status_window.pack()

#     # Create the flight log
#     self.flight_log = tk.Listbox(self.master, width=40, height=10)
#     self.flight_log.pack()

#     # Add some fake data to the GUI
#     self.add_fake_data()

#   def add_fake_data(self):
#     # Add some fake data to the duration label
#     self.duration_label.config
