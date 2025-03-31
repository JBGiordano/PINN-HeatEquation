import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from IPython.display import clear_output
import sys
import os
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(base_dir, 'example', 'data'))
sys.path.append(os.path.join(base_dir, 'extra'))

#%%   read csv
df = pd.read_csv(os.path.join(base_dir, 'example', 'data', 'temperatures.csv'))
t = pd.read_csv(os.path.join(base_dir, 'example', 'data', 'times.csv'))
t = t-820469098 #correct times :)

#%%   Neumann boundry condition
V = 23.8  # Voltage in volts
R = 11.1  # Resistance in ohms
dT_dx = 2.93  # Temperature gradient in °C/cm
L = 50  # Length of the rod in cm
#Calculate power
P = (V ** 2) / R
#Calculate ΔT
delta_T = dT_dx * L  # °C
#Calculate alpha (thermal diffusivity)
alpha = P / (R * delta_T)  # in cm²/s

#%%   Positions where the thermocouples measurements are taken.
x0 = np.array([8.14, 12.31, 16.41, 21.13, 24.96, 41.05])

#%%   Generate the PINN
k = 1.6      #Estimated value of \kappa to start training.
class MLP2(torch.nn.Module):

    def __init__(self,sizes):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.ka = torch.nn.Parameter(data=torch.Tensor([k]), requires_grad=True)
        for i in range(len(sizes)-1):
            self.layers.append(torch.nn.Linear(sizes[i],sizes[i+1]))
    def forward(self,x):
        h = x
        for hidden in self.layers[:-1]:
            h = torch.tanh(hidden(h))
        output = self.layers[-1]
        y = output(h)
        return y
    
#%%  Boundary conditions
x_data_cc50 = (50* torch.tensor( torch.ones(100))).view(-1,1) .requires_grad_(True)#100 posiciones para x = 0, 100 para x = 2pi
t_data_cc50 = torch.linspace(0,1,100).view(-1,1).requires_grad_(True)
input_cc50 = torch.cat((x_data_cc50.flatten().view(-1,1), t_data_cc50.flatten().view(-1,1)), dim=1)

x_data_cc0 = (50* torch.tensor(torch.zeros(100))).view(-1,1) .requires_grad_(True)
t_data_cc0 = torch.linspace(0,1,100).view(-1,1).requires_grad_(True)
input_cc0 = torch.cat((x_data_cc0.flatten().view(-1,1), t_data_cc0.flatten().view(-1,1)), dim=1)

#%%  Physics to train
t_physics = torch.linspace(0,1,200).requires_grad_(True)
x_physics = torch.linspace(0,50, 50).requires_grad_(True)
x_grid, t_grid = torch.meshgrid(x_physics, t_physics, indexing='ij')
x_grid = x_grid[:,:,None].requires_grad_(True) # add a dimension at the end so it can be used as input for the network
t_grid = t_grid[:,:,None].requires_grad_(True) 
input_physics = torch.cat((x_grid, t_grid), dim=-1)

#%%   Select a subset of elements to train the network (it can be done with all of them)
x1, t1 = df.iloc[0, 2:1721][::30], t.iloc[0, 2:1721][::30]
x2, t2 = df.iloc[1, 2:1721][::30], t.iloc[1, 2:1721][::30]
x3, t3 = df.iloc[2, 2:1721][::30], t.iloc[2, 2:1721][::30]
x4, t4 = df.iloc[3, 2:1721][::30], t.iloc[3, 2:1721][::30]
x5, t5 = df.iloc[4, 2:1721][::30], t.iloc[4, 2:1721][::30]
x6, t6 = df.iloc[5, 2:1721][::30], t.iloc[5, 2:1721][::30]

#%%
u_data = []
for x in [x1, x2, x3, x4, x5]:
    u_data.append(x.tolist())
u_data = torch.tensor(u_data).requires_grad_(True).view(-1,1)
t_data = [torch.from_numpy(t.to_numpy()).view(-1, 1).float() for t in [t1, t2, t3, t4, t5]]
t_data_nor = [t / 8253 for t in t_data]
x0 = torch.tensor([8.14, 12.31, 16.41, 21.13, 24.96])
#x0_exp = [x0_data[i].expand(t.shape) for i, t in enumerate(t_data_nor)]

#%%   Since the times of the thermocouples are the same, I take just one (t4), normalize it, and create a meshgrid.
t = torch.from_numpy((t4/ 8253).to_numpy()).float()
x_data_grid, t_data_grid = torch.meshgrid(x0,t,indexing= 'ij')
x_data_grid = x_data_grid[:,:,None]
t_data_grid = t_data_grid[:,:,None]
input_data = torch.cat((x_data_grid, t_data_grid), dim=-1)

#%%   Generate the PINN
pinn2 = MLP2([2,64,64,64, 64,1])

#%%   import the PINN if it is saved; otherwise, it starts running from scratch.
pinn2.load_state_dict(torch.load(os.path.join(base_dir, 'extra', 'pinn2_loss_perf_1e-5.pth')))
pinn2.eval()

#%%   Import loss if it is saved
with open(os.path.join(base_dir, 'extra', 'loss_perf_1e-5.txt'), 'r') as file:
    list_ = file.readlines()
loss_plt = [line.strip() for line in list_]

#%%   define a list to store the errors and then plot them.
loss_plt = []

#%%   separate the optimizer to modify the learning rate
optimizer = torch.optim.Adam(pinn2.parameters(),lr=1e-5)

#%%   Training + plotting of the error on a logarithmic scale every 100 iterations.
plt.ion()  
iterations =  10000
l =    1e-5
lc = 1e-4
p = (23-8)**2/(11.1*1.6*8634)
for epoch in range(iterations):
    optimizer.zero_grad()
    u_pred = pinn2(input_data).view(-1,1)
    loss1 = torch.mean((u_pred - u_data)**2)

    u_cc50 = pinn2(input_cc50)
    u_cc0 = pinn2(input_cc0)
    loss3 = lc * torch.mean(torch.autograd.grad(u_cc50, x_data_cc50, torch.ones_like(u_cc50), create_graph=True)[0] ** 2)
    loss4 = lc * torch.mean(( torch.autograd.grad(u_cc0, x_data_cc0, torch.ones_like(u_cc0), create_graph=True)[0]+ p )**2)
   
    yhp = pinn2(input_physics)
    dx  = torch.autograd.grad(yhp, x_grid, torch.ones_like(yhp), create_graph=True)[0]# compute u_x
    dx2  = torch.autograd.grad(dx, x_grid, torch.ones_like(yhp), create_graph=True)[0]# compute u_xx
    dt  = torch.autograd.grad(yhp, t_grid, torch.ones_like(yhp), create_graph=True)[0]# compute u_t
    physics = pinn2.ka*8324*dx2-dt
    
    loss2 =  l * torch.mean(physics**2)  #Weighted MSE error by l
    #loss = loss1 + loss2 + loss3  #sum all the errors
    loss =  loss2 + loss1 + loss4 + loss3
    #loss_list = [loss1,loss2,loss4,loss3]
    #loss_plt.append(loss_list)
    
    loss.backward()
    
    optimizer.step()
    
    with torch.autograd.no_grad():
        
        loss_plt.append([float(loss1), float(loss2), float(loss3+loss4)])
        #print(epoch,'Data',float(loss1),  'Física:',float(loss2),'CC (0):',float(loss4),'CC (50):', float(loss3), "Traning Loss:",float(loss.data))
        if epoch % 100 == 0:
            
            clear_output(wait=True)  # Clear the current output to avoid stacking plots

            fig, ax = plt.subplots() 
            x_values = [sublist[0] for sublist in loss_plt]  # "Data"
            y_values = [sublist[1] for sublist in loss_plt]  # "Physics"
            z_values = [sublist[2] for sublist in loss_plt]  # "Boundary"

            # Plot the data
            ax.plot(range(len(loss_plt)), y_values,  label="Physics")
            ax.plot(range(len(loss_plt)), x_values,  label="Data")
            ax.plot(range(len(loss_plt)), z_values,  label="Boundary")
            ax.set_ylabel('Loss')
            ax.set_xlabel('Iteration')
            ax.set_yscale('log')
            ax.grid(True)
            ax.legend()
            plt.show()
            
#%%   Error as a function of the iteration
x_values = [sublist[0] for sublist in loss_plt]
y_values = [sublist[1] for sublist in loss_plt]
plt.plot(range(len(loss_plt)), y_values, label="Physics")  
plt.plot(range(len(loss_plt)), x_values, label="Data")  
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.yscale('log')
plt.grid(True)
plt.legend()
plt.show()

#%%   Save the loss in a .txt file
with open('loss_sad_1e-1.txt', 'w') as file:
    for sublist in loss_plt:
        file.write(' '.join(map(str, sublist)) + '\n')

#%%   Heat map
u_net = pinn2(input_physics).detach().numpy().flatten()
u_net = u_net.reshape(len(t_grid), len(x_grid))
plt.figure(figsize=(10, 6))
plt.imshow(u_net, aspect='auto', cmap='magma', extent=[t_grid.detach().numpy().min(), t_grid.detach().numpy().max(), x_grid.detach().numpy().min(), x_grid.detach().numpy().max()], origin='lower')
plt.colorbar(label='Temperatures')
plt.xlabel('Times (t_net)', fontsize=14)
plt.savefig('-.png', bbox_inches='tight')

#%%   SAVE PINN
torch.save(pinn2.state_dict(), 'pinn2_loss_bad_1e-2.pth')
