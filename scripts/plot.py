import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker, cm

current_dir = os.path.dirname(__file__);
plt.rcParams["savefig.directory"] = current_dir;

plt.figure(figsize=(30, 12))

for mode in range(0,2):
  data = np.genfromtxt('../build/release/intensity_mode' + str(mode) + '.txt', delimiter=',');
  print(data.shape)
  Nz = data.shape[0];
  Nt = data.shape[1];

  tvec = np.linspace(-40.0, 40.0, Nt);
  zvec = np.linspace(0.0, 7.5, Nz);
  tmat, zmat = np.meshgrid(tvec, zvec);

  # log_data = np.log10(np.clip(data, 1e-16, None));
  log_data = np.clip(20*np.log10(data), -50, None);

  print(np.max(np.abs(data[:])))

  plt.subplot(2, 2, mode + 1);
  cs = plt.contourf(tmat, zmat, log_data, 100); #, cmap ="bone")
  cbar = plt.colorbar(cs)
  plt.title('Mode ' + str(mode) + ' intensity')

  plt.subplot(2, 2, mode + 3);
  plt.plot(tvec,data[Nz-1,:]);
  
plt.show()

# plt.plot(data);
# plt.show()