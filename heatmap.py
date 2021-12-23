

from mpi4py import MPI
import numpy as np
import cv2
import math
from numpy import asarray
from PIL import Image
from scipy import signal
import PIL
import sys
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
t1 = MPI.Wtime()

iter = int(sys.argv[1])

t1 = MPI.Wtime()
num_alive = []

def init(size_x=100, size_y=100, coeff=0.8):
    data=np.random.choice([0, 1], (size_x, size_y), replace=True, p=[coeff, 1.0-coeff])
    return data

def plotheatmap(u_k, k):
  # Clear the current plot figure
  plt.clf()
  plt.title(f"Temperature at t = {k*delta_t:.3f} unit time")
  plt.xlabel("x")
  plt.ylabel("y")
  
  # This is to plot u_k (u at time-step k)
  plt.pcolormesh(u_k, cmap=plt.cm.jet, vmin=0, vmax=100)
  plt.colorbar()
  
  return plt

def animate(k):
  plotheatmap(grid[k], k)


alpha = 2
delta_x = 1
delta_t = (delta_x ** 2)/(4 * alpha)
gamma = (alpha * delta_t) / (delta_x ** 2)

plate_length = 100

def calculate(u):
    res = np.zeros((u.shape[0] - 2, u.shape[1] - 2))
    for i in range(1, u.shape[0]-1, delta_x):
        for j in range(1, u.shape[1] - 1, delta_x):
            res[i-1, j-1] = gamma * (u[i+1][j] + u[i-1][j] + u[i][j+1] + u[i][j-1] - 4*u[i][j]) + u[i][j]

    return res

def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 1)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value

def do_animation(k):
        #print(f'k: {k}')
        return plotheatmap(grids[k], k)

global grid
grid = []
for it in range(iter):
    
    #print(it)
    if rank == 0:
        if it == 0:
          plate_length = 100
          max_iter_time = 100

          # Initialize solution: the grid of u(k, i, j)
          data = np.empty((plate_length, plate_length))

          # Initial condition everywhere inside the grid
          u_initial = 0

          # Boundary conditions
          u_top = 100.0
          u_left = 100.0
          u_bottom = 100.0
          u_right = 100.0

          # Set the initial condition
          data.fill(u_initial)

          # Set the boundary conditions
          data[(plate_length-1):, :] = u_top
          data[:, :1] = u_left
          data[:1, 1:] = u_bottom
          data[:, (plate_length-1):] = u_right
          #grid = []
        else:
          data = grid[-1]
    else:
        data = None

    data_loc = comm.bcast(data, root=0)
    
    if size == 1:
        data_received = calculate(data_loc)

    if rank == 0:
      data_loc = data_loc[int(data_loc.shape[0] / size) * rank : int(data_loc.shape[0] / size) * (rank + 1) + 1, :]
      data_received = calculate(data_loc)
    elif rank == size - 1:
      data_loc = data_loc[int(data_loc.shape[0] / size) * rank - 1 : int(data_loc.shape[0] / size) * (rank + 1), :]
      data_received = calculate(data_loc)
    else: 
      data_loc = data_loc[int(data_loc.shape[0] / size) * rank - 1 : int(data_loc.shape[0] / size) * (rank + 1) + 1, :]
      data_received = calculate(data_loc)

    data_res = comm.gather(data_received,root=0)
    res = []
    # if rank == 0: 
    #   res = np.array(data_res[0])
    #   for j in range(1, size):
    #       res = np.vstack((res, data_res[j]))
    #   #print(data) 
    #   res = np.pad(res, 1, pad_with)
    #   grid.append(res)

      #print(res)
      
      """fig, ax = plt.subplots(figsize=(10, 10))
      plt.imshow(res, cmap='gist_heat')
      plt.savefig(f'/content/drive/MyDrive/skoltech/hppl/project/prj/prj_{100+it}.png')"""

      #print(grid[-1])
      
if rank == 0:
    """print('anim in process')
    anim = animation.FuncAnimation(plt.figure(), animate, interval=1, frames=max_iter_time, repeat=False)
    anim.save('heat_equation_solution.mp4')"""
    end = MPI.Wtime()
    print(float(end - t1))