

from mpi4py import MPI
import numpy as np
import cv2
import math
from numpy import asarray
from PIL import Image
from scipy import signal
import PIL
import matplotlib.pyplot as plt
import matplotlib as ml

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
t1 = MPI.Wtime()

iter = 300

t1 = MPI.Wtime()

def boundary(grid):
    x = np.linspace(0,1,len(grid))
    
    grid[0,:]  = np.interp(x,[0,1],[0,1])
    grid[:,-1] = np.interp(x,[0,1],[1,0])
    grid[-1,:] = np.interp(x,[0,1],[-1,0])
    grid[:,0]  = np.interp(x,[0,1],[0,-1])

def initgrid(gridsize):
    x = np.random.randn(gridsize,gridsize)
    boundary(x)
    return x

def plot(data):
    figure(figsize=(50,50))
    imshow(data, aspect='auto', interpolation='none')
    gca().set_xticks([])
    gca().set_yticks([])

def showsol(sol):
    plt.imshow(sol.T,cmap=ml.cm.Blues,interpolation='none',origin='lower')

def update(filter, old_data):
    neibs = signal.convolve2d(old_data, filter, mode='same', boundary='wrap')
    lives = neibs * old_data
    deads = neibs * (1 - old_data)
    data = np.where(np.where((lives < 4), lives, 0) > 1, lives, 0) + np.where(deads == 3, deads, 0)
    data[data > 1] = 1
    return data

def gauss_seidel(grid):
    newgrid = grid.copy()
    
    for i in range(1,newgrid.shape[0]-1):
        for j in range(1,newgrid.shape[1]-1):
            newgrid[i,j] = 0.25 * (newgrid[i,j+1] + newgrid[i,j-1] +
                                   newgrid[i+1,j] + newgrid[i-1,j])
    
    return newgrid

#if rank == 0:
#    data = initgrid(25)
#else:
#    data = None

for it in range(iter):
    if rank == 0:
        if it == 0:
            data = initgrid(400)
        else:
          data = res
    else:
        data = None


    data_loc = comm.bcast(data, root=0)
    data_loc = np.array(data_loc)
    #print('data_loc ', data_loc.shape)
    
    if size == 1:
        data_received = gauss_seidel(data_loc)
    else:
        if rank == 0:
          data_loc = data_loc[int(data_loc.shape[0] / size) * rank : int(data_loc.shape[0] / size) * (rank + 1) + 1, :]
          data_received = gauss_seidel(data_loc)[:-1]
        elif rank == size - 1:
          data_loc = data_loc[int(data_loc.shape[0] / size) * rank - 1 : int(data_loc.shape[0] / size) * (rank + 1), :]
          data_received = gauss_seidel(data_loc)[1:]
        else: 
          data_loc = data_loc[int(data_loc.shape[0] / size) * rank - 1 : int(data_loc.shape[0] / size) * (rank + 1) + 1, :]
          data_received = gauss_seidel(data_loc)[1:-1]

    data = comm.gather(data_received,root=0)


    # res = []
    # if rank == 0:
    #   #print(it)
    #   #print(data[0].shape)
    #   res = np.array(data[0])
    #   for j in range(1, size):
    #       res = np.vstack((res, data[j]))
    #       #print(data[j].shape)
    #       #print('res ', res.shape)
      
    #   #print(res)
    #   #fig, ax = plt.subplots(figsize=(16, 16))
    #   #plt.imshow(res, cmap="gray")
    #   #plt.savefig(f'/content/drive/MyDrive/skoltech/hppl/life_beacon/life{it+1}.png')
    #   plt.figure(figsize=(12,12))
    #   if it % 20 == 0:
    #       showsol(res)
    #       plt.title('iter = %s' % it)
    #       plt.savefig(f'/content/drive/MyDrive/Учёба/проект/gauss_seidel/gauss_seidel{it+1}.png')

    #print(res)
if rank == 0:
    end = MPI.Wtime()
    print(float(end - t1))
