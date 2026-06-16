import numpy as np
input_shape = (20, 11)
payload = [[6000.0, 6050.0, 5950.0, 6020.0, 1000000, 6050.0, 50.0, 0.0, 0.0, 15.0, 0.05]] * 20
x = np.array(payload, dtype=np.float32)

x_2d = x.reshape(-1, input_shape[1])
col_min = np.min(x_2d, axis=0)
col_max = np.max(x_2d, axis=0)
range_vals = np.where((col_max - col_min) == 0, 1, col_max - col_min)
x_scaled = (x_2d - col_min) / range_vals
x = x_scaled.reshape(1, *input_shape)

print(x.flatten())
