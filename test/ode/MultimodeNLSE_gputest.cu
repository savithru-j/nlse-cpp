// autodiffeq - C++ library for sensitivity analysis of ODEs
// Licensed under the MIT License (http://opensource.org/licenses/MIT)
// Copyright (c) 2023, Savithru Jayasinghe

#include <gtest/gtest.h>
#include "ode/MultimodeNLSE.hpp"
#include "ode/GPUMultimodeNLSE.hpp"

using namespace autodiffeq;

//----------------------------------------------------------------------------//
TEST( MultimodeNLSE_2mode, EvalRHS_CPU_GPU_Consistency )
{
  using Complex = complex<double>;

  const int num_modes = 2;
  const int num_time_points = 129;

  Array2D<double> beta_mat = 
    {{ 0.00000000e+00, -5.31830434e+03},
     { 0.00000000e+00,  1.19405403e-01},
     {-2.80794698e-02, -2.82196091e-02},
     { 1.51681819e-04,  1.52043264e-04},
     {-4.95686317e-07, -4.97023237e-07}};

  Array4D<double> Sk(2,2,2,2, {4.9840660e+09, 0.0000000e+00, 0.0000000e+00, 2.5202004e+09,
                               0.0000000e+00, 2.5202004e+09, 2.5202004e+09, 0.0000000e+00,
                               0.0000000e+00, 2.5202004e+09, 2.5202004e+09, 0.0000000e+00,
                               2.5202004e+09, 0.0000000e+00, 0.0000000e+00, 3.7860385e+09});

  double tmin = -20, tmax = 20;
  double n2 = 2.3e-20;
  double omega0 = 1.2153e3;
  bool is_self_steepening = false;
  bool is_nonlinear = true;

  MultimodeNLSE<Complex> ode_cpu(num_modes, num_time_points, tmin, tmax, beta_mat,
                                 n2, omega0, Sk, is_self_steepening, is_nonlinear);

  const int sol_size = num_modes * num_time_points;
  Array1D<Complex> sol_cpu(sol_size);
  for (int i = 0; i < sol_size; ++i)
    sol_cpu(i) = {0.2*i, -0.01*i*i}; //Initialize to some non-zero solution
  
  int step = 0;
  double z = 0.0;
  Array1D<Complex> rhs_cpu(sol_size);
  ode_cpu.EvalRHS(sol_cpu, step, z, rhs_cpu);

  GPUMultimodeNLSE<Complex> ode_gpu(num_modes, num_time_points, tmin, tmax, beta_mat,
                                    n2, omega0, Sk, is_self_steepening, is_nonlinear);

  //TODO: CPU and GPU have different data orderings, so need to copy elements to transformed indices
  Array1D<Complex> sol_cpu_colmajor(sol_size);
  for (int t = 0; t < num_time_points; ++t)
    for (int p = 0; p < num_modes; ++p)
      sol_cpu_colmajor(t*num_modes + p) = sol_cpu(p*num_time_points + t);

  GPUArray1D<Complex> sol_gpu(sol_cpu_colmajor);
  GPUArray1D<Complex> rhs_gpu(sol_size);
  ode_gpu.EvalRHS(sol_gpu, step, z, rhs_gpu);
  auto rhs_gpu_host = rhs_gpu.CopyToHost();

  //Check the consistency of CPU and GPU residuals
  for (int t = 0; t < num_time_points; ++t)
    for (int p = 0; p < num_modes; ++p)
    {
      const auto& r_cpu = rhs_cpu(p*num_time_points + t);
      const auto& r_gpu = rhs_gpu_host(t*num_modes + p);
      EXPECT_DOUBLE_EQ(r_cpu.real(), r_gpu.real());
      EXPECT_DOUBLE_EQ(r_cpu.imag(), r_gpu.imag());
    }
}
