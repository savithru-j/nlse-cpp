import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker, cm
import h5py

current_dir = os.path.dirname(__file__);
plt.rcParams["savefig.directory"] = current_dir;

# f_mpa = h5py.File('GRIN_1550_linear_sample_single_gpu_mpa.mat','r')
f_mpa = h5py.File('GRIN_1550_nofR_noSS_c0_sample_single_gpu_mpa.mat','r')
# data = scipy.io.loadmat('GRIN_1550_linear_sample_single_gpu_mpa.mat')
print(f_mpa.keys())
field_data = np.array(f_mpa.get('prop_output/fields'));
complex_field_data = field_data['real'] + 1j*field_data['imag']
print(np.shape(complex_field_data))

plt.figure(figsize=(30, 12))
plt.rcParams.update({'font.size': 16})

zstep_mpa = 15000; #10mm
zstep_cpp = 150;

for mode in range(0,2):
  # cpp_data = np.genfromtxt('../build/release/intensity_mode' + str(mode) + '_rk4_tderivorder2_nz3e5_gpu.txt', delimiter=',');
  cpp_data = np.genfromtxt('../build/release/examples/intensity_mode' + str(mode) + '.txt', delimiter=',');
  # cpp_data = np.genfromtxt('solutions/intensity_mode' + str(mode) + '_kerr_noSS_tderivorder4_nz3e5_nt8193_cpu.txt', delimiter=',');
  print("cpp_data shape:", cpp_data.shape)
  Nt_cpp = cpp_data.shape[1];
  abs_u_cpp = np.transpose(cpp_data[zstep_cpp,:]);
  intensity_db_cpp = np.maximum(20*np.log10(abs_u_cpp), -50);
  print(abs_u_cpp.shape)

  u_mpa = np.transpose(complex_field_data[zstep_mpa,mode,:]);
  print(u_mpa.shape)
  
  abs_u_mpa = np.abs(u_mpa);
  intensity_db_mpa = np.maximum(20*np.log10(abs_u_mpa), -50);
  Nt_mpa = u_mpa.shape[0];

  tvec_mpa = np.linspace(-40.0, 40.0, Nt_mpa);
  print(tvec_mpa[1::].shape)

  tvec_cpp = np.linspace(-40.0, 40.0, Nt_cpp);
  # print(tvec[-1])
  # tvec2 = np.concatenate([tvec[1::].reshape(Nt-1,1),tvec[-1].reshape(1,1)]);
  # print(tvec2.shape)

  # log_data = np.log10(np.clip(abs_u0_mpa, 1e-16, None));

  print('max_abs_mpa: ', np.max(abs_u_mpa))
  print('max_abs_cpp: ', np.max(abs_u_cpp))
  # print('max_diff: ', np.max(np.abs(abs_u_cpp - abs_u_mpa)))

  plt.subplot(1, 2, mode + 1);
  # plt.plot(tvec_mpa, abs_u_mpa, 'b-', tvec_cpp, abs_u_cpp, 'r-');
  plt.plot(tvec_mpa, intensity_db_mpa, 'b-', tvec_cpp, intensity_db_cpp, 'r-');
  plt.legend(['MPA','Finite difference with RK4'], loc='upper left')
  plt.xlabel('Time [ps]')
  plt.ylabel('Intensity (dB)')
  
plt.show()
