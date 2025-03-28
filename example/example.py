import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from pypinn.train import train
import matplotlib.pyplot as plt
base_dir = os.path.dirname(__file__)
dfpath = os.path.join(base_dir, 'data', 'temperatures.csv')
tpath = os.path.join(base_dir, 'data', 'times.csv')

#%%   read csv
df = pd.read_csv(dfpath)
t = pd.read_csv(tpath)
t = t-820469098 #correct times :)

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
#   Positions where the thermocouples measurements are taken.
x0 = torch.tensor([8.14, 12.31, 16.41, 21.13, 24.96])
#x0_exp = [x0_data[i].expand(t.shape) for i, t in enumerate(t_data_nor)]
#%%   Since the times of the thermocouples are the same, I take just one (t4), normalize it, and create a meshgrid.
t = torch.from_numpy((t4/ 8253).to_numpy()).float()
x_data_grid, t_data_grid = torch.meshgrid(x0,t,indexing= 'ij')
x_data_grid = x_data_grid[:,:,None]
t_data_grid = t_data_grid[:,:,None]
input_data = torch.cat((x_data_grid, t_data_grid), dim=-1)

#%%
#Values to change during interrupts
it =  10000              #iterations
lr = 1e-5                #learning rate
l =    1e-5
lc = 1e-4
k = 1                    #estimed k

results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
path_pinn = os.path.join(results_dir, 'pinn.pth') if os.path.exists(os.path.join(results_dir, 'pinn.pth')) else None
path_loss = os.path.join(results_dir, 'loss.txt') if os.path.exists(os.path.join(results_dir, 'loss.txt')) else None

loss_plt, pinn = train(k, lr,iterations, [l,lc] ,input_data,u_data,
          input_cc50,input_cc0,x_data_cc50,
          x_data_cc0,input_physics,x_grid,t_grid,
          path_pinn=path_pinn, path_loss=path_loss)
    
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
#%%
u_net = pinn(input_physics).detach().numpy().flatten()
u_net = u_net.reshape(len(t_grid), len(x_grid))
plt.figure(figsize=(10, 6))
plt.imshow(u_net, aspect='auto', cmap='magma', extent=[t_grid.detach().numpy().min(), t_grid.detach().numpy().max(), x_grid.detach().numpy().min(), x_grid.detach().numpy().max()], origin='lower')
plt.colorbar(label='Temperatures')
plt.xlabel('Times (t_net)', fontsize=14)
plt.savefig('-.png', bbox_inches='tight')

