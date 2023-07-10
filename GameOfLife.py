import pygame
import numpy as np
from pygame import *
import time
from numba import cuda
import math as m

# pip install -r requirements.txt

class Cell:
    def __init__(self, pos_x, pos_y, radius, color_alive, color_dead):
        self.radius = radius
        self.color_alive = color_alive
        self.color_dead = color_dead
        self.color = (0,0,0) #init


    def DrawCell(self, etat, pos_x, pos_y):
        if etat == 1 :
            self.color = self.color_alive
        else:
            self.color = self.color_dead
        draw.circle(window, self.color, (pos_x + self.radius, pos_y + self.radius), self.radius)


    def Update(self, tab0):               
        #Drawing new frame
        CellSize_i = 0
        for i in range(m):
            CellSize_i = i*CELL_SIZE #avoid to recalculate it for each j
            for j in range(n):
                #tab0[i][j] = tab1[i][j]
                cellule.DrawCell(tab0[i][j], CellSize_i , j*CELL_SIZE)

@cuda.jit
def UpdateTabGPU(tab0, out):
    x,y = cuda.grid(2)
    count = 0
    for k in range(-1,2):
        for l in range(-1,2):
            if(tab0[x+k][y+l] == 1):
                count += 1

    if(tab0[x][y] == 1):
        count -= 1

    if(count == 2):
        out[x][y] = tab0[x][y]
    elif(count == 3):
        out[x][y] = 1
    else:
        out[x][y] = 0

def UpdateTabCPU(tab0):
    #Calculation of new state
    tab1 = np.zeros_like(tab0)
    for i in range(0,m-1):
        for j in range(0,n-1):
            count = 0
            for k in range(-1,2):
                for l in range(-1,2):
                    if(tab0[i+k][j+l] == 1):
                        count += 1

            if(tab0[i][j] == 1):
                count -= 1

            if(count == 2):
                tab1[i][j] = tab0[i][j]
            elif(count == 3):
                tab1[i][j] = 1
            else:
                tab1[i][j] = 0
    return tab1


if __name__ == '__main__':
    pygame.init()
    running = 1

    CELL_COLOR = (0,255,0)
    WIN_WIDTH = 1280
    WIN_HEIGHT = 720
    CELL_SIZE = 5
    BG_COLOR = (0, 0, 0) # couleur du fond
    flags = DOUBLEBUF

    print("Chose a resolution :")
    width = input(f"default WIDTH = {WIN_WIDTH} enter new width or press enter to use default width : \n")
    if(width != ""):
        WIN_WIDTH = int(width)         

    height = input(f"default HEIGHT = {WIN_HEIGHT} enter new height or press enter to use default height : \n")
    if(height != ""):
        WIN_HEIGHT = int(height)         

    print("Choose mode :\n1 = GPU\n2 = CPU\n3 = compare CPU/GPU\n")
    mode = False
    while not mode:
        MODE = input("Choose a mode (1,2 or 3):\n")
        mode = True
        try:
            MODE = int(MODE)
        except:
            print("error enter 1, 2 or 3")
            mode = False

    interation_compare = 0     
    if(MODE == 3):
        interation_compare = int(input("Compare : \nHow many iterations :\n"))

    m = int(WIN_WIDTH/CELL_SIZE)
    n = int(WIN_HEIGHT/CELL_SIZE)
    tab0 = np.random.randint(0,1 + 1, size=(m, n))  # m x n éléments

    print("running : press Q to exit")

    window = display.set_mode((WIN_WIDTH, WIN_HEIGHT), flags)
    pygame.display.set_caption("Game Of Life")
    window.fill(BG_COLOR)

    clock = pygame.time.Clock()
    cellule = Cell(0,0,CELL_SIZE/2, CELL_COLOR, BG_COLOR)

    #Init T=0
    for i in range(m):
        for j in range(n):
            cellule.DrawCell(tab0[i][j], i*CELL_SIZE , j*CELL_SIZE)

    pygame.display.flip()

    iteration = 0

    init_done = False
    compute_time = 0.0

    compute_cpu = 0.0
    compute_gpu = 0.0

    cpu_iter = 0
    gpu_iter = 0

    while running:
        
        pygame.mouse.set_visible(0)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = 0
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    print("exit : Q pressed")
                    running = 0

    
        start_time = time.time()

        if(MODE == 2): #CPU
            tab0 = UpdateTabCPU(tab0)

        if(MODE == 1): #GPU
            if(not init_done):
                threads_per_block = (4,4)
                blocks = (1,1)
                try:
                    blockspergrid_x = m.ceil(tab0.shape[0] / threads_per_block[0])
                    blockspergrid_y = m.ceil(tab0.shape[1] / threads_per_block[1])
                except:
                    blockspergrid_x = int(tab0.shape[0] / threads_per_block[0])
                    blockspergrid_y = int(tab0.shape[1] / threads_per_block[1])
                    blocks = (blockspergrid_x, blockspergrid_y)
                d_tab0 = cuda.to_device(tab0) # here to avoid reallocation
                d_out = cuda.to_device(tab0)
                init_done = True

            UpdateTabGPU[blocks, threads_per_block](d_tab0, d_out); cuda.synchronize()
            tab0 = d_out.copy_to_host()
            #swap pointers : d_out is new tab0 and old tab0 will be d_out  -> avoid useless copy of tab0 to device when function is called by using preallocated device memory
            temp = d_out
            d_out = d_tab0
            d_tab0 = temp


       
        if(MODE == 3):
            if(cpu_iter < interation_compare):
                tab0 = UpdateTabCPU(tab0)
                cpu_iter += 1
            else:
                if(gpu_iter < interation_compare):
                    if(not init_done):
                        threads_per_block = (4,4)
                        blocks = (1,1)
                        try:
                            blockspergrid_x = m.ceil(tab0.shape[0] / threads_per_block[0])
                            blockspergrid_y = m.ceil(tab0.shape[1] / threads_per_block[1])
                        except:
                            blockspergrid_x = int(tab0.shape[0] / threads_per_block[0])
                            blockspergrid_y = int(tab0.shape[1] / threads_per_block[1])
                            blocks = (blockspergrid_x, blockspergrid_y)
                        d_tab0 = cuda.to_device(tab0) # here to avoid reallocation
                        d_out = cuda.to_device(tab0)
                        init_done = True

                    UpdateTabGPU[blocks, threads_per_block](d_tab0, d_out); cuda.synchronize()
                    tab0 = d_out.copy_to_host()
                    #swap pointers : d_out is new tab0 and old tab0 will be d_out  -> avoid useless copy of tab0 to device when function is called by using preallocated device memory
                    temp = d_out
                    d_out = d_tab0
                    d_tab0 = temp
                    gpu_iter += 1
                else:
                    running = 0 #end of comparaison

        end_time = time.time()
        compute_time += end_time - start_time
        if(MODE == 3 and running != 0):
            if(cpu_iter < interation_compare):
                compute_cpu += end_time - start_time
            else:
                compute_gpu += end_time - start_time
        
        cellule.Update(tab0)
        
        pygame.display.flip() #change display buffer to the updated one
        iteration += 1
        clock.tick(60) # 60Hz to go BRRrrrr mode

    if(MODE == 1 or MODE == 2):
        print(f"average computation time : {compute_time/iteration} sec")
    else: #compare
        print(f"average CPU time : {compute_cpu/interation_compare} sec")
        print(f"average GPU time : {compute_gpu/interation_compare} sec")
        print(f"GPU is faster than CPU by {100*(compute_cpu-compute_gpu)/compute_cpu}%")