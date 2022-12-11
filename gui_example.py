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
   
