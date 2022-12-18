# Import the necessary libraries
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class RocketGui:
  def __init__(self, master):
    # Create the main window
    self.master = master
    self.master.geometry('600x600')
    self.master.title('Rocket GUI')

    # Create the duration label
    self.duration_label = tk.Label(self.master, text='Duration: 0 s')
    self.duration_label.pack()

    # Create the map frame
    self.map_frame = tk.Frame(self.master, width=400, height=400)
    self.map_frame.pack()

    # Create the map figure and canvas
    self.map_figure = Figure(figsize=(4, 4))
    self.map_canvas = FigureCanvasTkAgg(self.map_figure, self.map_frame)
    self.map_canvas.draw()
    self.map_canvas.get_tk_widget().pack()

    # Create the status window
    self.status_window = tk.Text(self.master)
    self.status_window.pack()

    # Create the flight log
    self.flight_log = []

  def draw_trajectory(self, trajectory):
    # Clear the map figure
    self.map_figure.clf()

    # Draw the trajectory on the map figure
    ax = self.map_figure.add_subplot(111)
    ax.plot(trajectory[:, 0], trajectory[:, 1])

    # Redraw the map canvas
    self.map_canvas.draw()

  def update_status(self, status):
    # Append the status to the flight log
    self.flight_log.append(status)

    # Clear the status window
    self.status_window.delete(1.0, tk.END)

    # Update the status window with the flight log
    for log in self.flight_log:
      self.status_window.insert(tk.END, log)

  def update_duration(self, duration):
    # Update the duration label
    self.duration_label.configure(text='Duration: {} s'.format(duration))

# ---------------------
# import sys
# from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QLineEdit, QPushButton

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

#         # Set widget positions
#         self.label.move(20, 20)
#         self.line_edit.move(20, 50)
#         self.button.move(20, 80)

#         # Connect button click to function
#         self.button.clicked.connect(self.predictTrajectory)

#         # Set window properties
#         self.setGeometry(300, 300, 300, 200)
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
