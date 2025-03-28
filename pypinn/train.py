import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from IPython.display import clear_output
from pypinn.mlp import MLP
import matplotlib.pyplot as plt

def save_files(pinn, loss_plt, base_dir=None, prefix='pinn', loss_filename='loss.txt'):
    if base_dir is None:
        base_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        
    model_path = os.path.join(base_dir, f'{prefix}.pth')
    counter = 1
    while os.path.exists(model_path):
        model_path = os.path.join(base_dir, f'{prefix}{counter}.pth')
        counter += 1
    
    torch.save(pinn.state_dict(), model_path)
    
    loss_path = os.path.join(base_dir, loss_filename)
    with open(loss_path, 'w') as file:
        for sublist in loss_plt:
            file.write(' '.join(map(str, sublist)) + '\n')

def train(k,lr,iterations, ls ,input_data,u_data,
          input_cc50,input_cc0,x_data_cc50,
          x_data_cc0,input_physics,x_grid,t_grid,
          path_pinn = None, path_loss = None ):
    if path_pinn == None:
        pinn = MLP([2,64,64,64, 64,1],k)
    else:
        pinn = MLP([2,64,64,64, 64,1],k)
        pinn.load_state_dict(torch.load(path_pinn))
        pinn.eval()
    
    if path_loss == None:
        loss_plt = []
    else:
        with open(path_loss, 'r') as file:
            list_ = file.readlines()
        loss_plt = [line.strip() for line in list_]
    
    optimizer = torch.optim.Adam(pinn.parameters(),lr)
    
    plt.ion()  
    p = (23-8)**2/(11.1*1.6*8634)
    try:
        for epoch in range(iterations):
            optimizer.zero_grad()
            u_pred = pinn(input_data).view(-1,1)
            loss1 = torch.mean((u_pred - u_data)**2)
    
            u_cc50 = pinn(input_cc50)
            u_cc0 = pinn(input_cc0)
            loss3 = ls[1] * torch.mean(torch.autograd.grad(u_cc50, x_data_cc50, torch.ones_like(u_cc50), create_graph=True)[0] ** 2)
            loss4 = ls[1] * torch.mean(( torch.autograd.grad(u_cc0, x_data_cc0, torch.ones_like(u_cc0), create_graph=True)[0]+ p )**2)
           
            yhp = pinn(input_physics)
            dx  = torch.autograd.grad(yhp, x_grid, torch.ones_like(yhp), create_graph=True)[0]# compute u_x
            dx2  = torch.autograd.grad(dx, x_grid, torch.ones_like(yhp), create_graph=True)[0]# compute u_xx
            dt  = torch.autograd.grad(yhp, t_grid, torch.ones_like(yhp), create_graph=True)[0]# compute u_t
            physics = pinn.ka*8324*dx2-dt
            
            loss2 =  ls[0] * torch.mean(physics**2)  #Weighted MSE error by l
            loss =  loss2 + loss1 + loss4 + loss3            
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
                    
                    ax.plot(range(len(loss_plt)), y_values,  label="Physics")
                    ax.plot(range(len(loss_plt)), x_values,  label="Data")
                    ax.plot(range(len(loss_plt)), z_values,  label="Boundary")
                    ax.set_ylabel('Loss')
                    ax.set_xlabel('Iteration')
                    ax.set_yscale('log')
                    ax.grid(True)
                    ax.legend()
                    plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        save_files(pinn, loss_plt)
    return loss_plt, pinn
