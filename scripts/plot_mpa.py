import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker, cm
# import scipy.io
import h5py


f = h5py.File('GRIN_1550_linear_sample_single_gpu_mpa.mat','r')
# data = scipy.io.loadmat('GRIN_1550_linear_sample_single_gpu_mpa.mat')
print(f.keys())
# data = f.get('prop_output/fields')
field_data = np.array(f.get('prop_output/fields'));
complex_field_data = field_data['real'] + 1j*field_data['imag']
print(np.shape(complex_field_data))

plt.figure(figsize=(30, 12))

for mode in range(0,2):
  # data = np.genfromtxt('../build/release/intensity_mode' + str(mode) + '.txt', delimiter=',');

  data_mpa = np.transpose(complex_field_data[:,mode,:]);
  print(data_mpa.shape)
  
  abs_data_mpa = np.abs(data_mpa);
  Nt = data_mpa.shape[0];
  Nz = data_mpa.shape[1];

  tvec = np.linspace(-40.0, 40.0, Nt);
  zvec = np.linspace(0.0, 0.1, Nz);
  tmat, zmat = np.meshgrid(tvec, zvec);

  # log_data = np.log10(np.clip(abs_u0_mpa, 1e-16, None));

  print(np.max(abs_data_mpa))

  plt.subplot(1, 2, mode + 1);
  # plt.plot(tvec, abs_u0_mpa);
  cs = plt.contourf(tmat, zmat, np.transpose(abs_data_mpa), 100); #, cmap ="bone")
  cbar = plt.colorbar(cs)
  plt.title('Mode ' + str(mode) + ' intensity')
  
plt.show()
