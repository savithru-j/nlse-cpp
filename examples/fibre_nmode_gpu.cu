// nlse++
// C++ library for solving the nonlinear Schrödinger equation
// Licensed under the MIT License (http://opensource.org/licenses/MIT)
// Copyright (c) 2023, Savithru Jayasinghe

#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <autodiffeq/numerics/ADVar.hpp>
#include <autodiffeq/numerics/Complex.hpp>
#include <autodiffeq/solver/RungeKutta.hpp>

#include "ode/GPUMultimodeNLSE.hpp"
#include "ode/InitialCondition.hpp"

using namespace autodiffeq;

int main()
{
  using Complex = complex<double>;
  using clock = std::chrono::high_resolution_clock;

  const int num_modes = 2;
  const int num_time_points = 8193;

  //Read beta values from 5x30 matrix stored in file
  const int max_Ap_tderiv = 4;
  Array2D<double> beta_mat(max_Ap_tderiv+1, num_modes);
  {
    std::ifstream f_beta("data/beta5x30.txt");
    if (!f_beta.is_open()) {
      std::cout << "Could not open beta data file!" << std::endl;
      return 1;
    }
    Array1D<double> beta_values;
    double tmp_val;
    while (f_beta >> tmp_val)
      beta_values.push_back(tmp_val);
    f_beta.close();
    Array2D<double> beta_mat_5x30(5, 30, beta_values.GetDataVector());
    beta_values.clear();

    for (int i = 0; i <= max_Ap_tderiv; ++i)
    {
      double unit_conv_factor = std::pow(1000,(1.0-i));
      for (int j = 0; j < num_modes; ++j)
        beta_mat(i,j) = beta_mat_5x30(i,j) * unit_conv_factor;
    }
  }
  std::cout << "Loaded beta data from file." << std::endl;
  // std::cout << std::scientific << std::setprecision(8);
  // std::cout << beta_mat << std::endl;

  //Read Sk tensor values from 20x20x20x20 matrix stored in file
  Array4D<double> Sk(num_modes, num_modes, num_modes, num_modes);
  {
    std::ifstream f_Sk("data/Sk_tensor_20modes.txt");
    if (!f_Sk.is_open()) {
      std::cout << "Could not open Sk tensor data file!" << std::endl;
      return 1;
    }
    Array1D<double> Sk_values;
    double tmp_val;
    while (f_Sk >> tmp_val)
      Sk_values.push_back(tmp_val);
    f_Sk.close();
    Array4D<double> Sk_20mode(20, 20, 20, 20, Sk_values.GetDataVector());
    Sk_values.clear();

    for (int i = 0; i < num_modes; ++i)
      for (int j = 0; j < num_modes; ++j)
        for (int k = 0; k < num_modes; ++k)
          for (int l = 0; l < num_modes; ++l)
            Sk(i,j,k,l) = Sk_20mode(i,j,k,l);
  }
  std::cout << "Loaded Sk tensor data from file." << std::endl;
  // std::cout << Sk << std::endl;

  double tmin = -40, tmax = 40;
  double n2 = 2.3e-20;
  double omega0 = 1.2153e3;
  bool is_self_steepening = false;
  bool is_nonlinear = true;

  GPUMultimodeNLSE<Complex> ode(num_modes, num_time_points, tmin, tmax, beta_mat,
                                n2, omega0, Sk, is_self_steepening, is_nonlinear);

  Array1D<double> Et(num_modes, 10.0); //nJ (in range [6,30] nJ)
  Array1D<double> t_FWHM(num_modes, 0.1); //ps (in range [0.05, 0.5] ps)
  Array1D<double> t_center(num_modes, 0.0); //ps
  Array1D<Complex> sol0;
  ComputeGaussianPulse(Et, t_FWHM, t_center, ode.GetTimeVector(), sol0);

  double z_start = 0, z_end = 7.5; //[m]
  int nz = 15000*20;
  int storage_stride = 100*20;

  // double z_start = 0, z_end = 1.0; //[m]
  // int nz = 2000*20;
  // int storage_stride = 100*20;

  std::cout << "Problem parameters:\n"
            << "  No. of modes          : " << num_modes << "\n"
            << "  Time range            : [" << tmin << ", " << tmax << "] ps\n"
            << "  Z max                 : " << z_end << " m\n"
            << "  No. of time points    : " << num_time_points << "\n"
            << "  No. of z-steps        : " << nz << "\n"
            << "  Solution storage freq.: Every " << storage_stride << " steps\n" 
            << std::endl;

  const int order = 4;
  RungeKutta<Complex> solver(ode, order);
  solver.SetSolveOnGPU(true);

  std::cout << "Solving ODE..." << std::endl;
  auto t0 = clock::now();
  auto sol_hist = solver.Solve(sol0, z_start, z_end, nz, storage_stride);
  auto t1 = clock::now();
  auto time_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count() / 1000.0;
  std::cout << std::fixed << std::setprecision(3);
  std::cout << "Solved in " << time_elapsed << " secs." << std::endl;
  std::cout << "Solution history size: " << sol_hist.GetNumStepsStored() 
            << " steps stored, solution dim: " << sol_hist.GetSolutionSize() << std::endl;

  for (int mode = 0; mode < num_modes; ++mode)
  {
    std::string filename = "intensity_mode" + std::to_string(mode) + ".txt";
    std::cout << "Writing solution file: " << filename << std::endl;
    std::ofstream f(filename, std::ios_base::out | std::ios::binary);
    f << std::setprecision(6) << std::scientific;
    for (int i = 0; i < sol_hist.GetNumSteps(); ++i)
    {
      if (i % storage_stride == 0)
      {
        for (int j = 0; j < num_time_points-1; ++j)
          f << abs(sol_hist(i, j*num_modes + mode)) << ", ";
        f << abs(sol_hist(i, (num_time_points-1)*num_modes + mode)) << std::endl;
      }
    }
    f.close();
  }
}