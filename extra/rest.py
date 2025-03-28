#   Neumann boundry condition
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