// nlse++
// C++ library for solving the nonlinear Schrödinger equation
// Licensed under the MIT License (http://opensource.org/licenses/MIT)
// Copyright (c) 2023, Savithru Jayasinghe

#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>

#ifdef ENABLE_OPENMP
#include <omp.h>
#endif

#include <autodiffeq/numerics/ADVar.hpp>
#include <autodiffeq/numerics/ADVarS.hpp>
#include <autodiffeq/numerics/Complex.hpp>
#include <autodiffeq/solver/RungeKutta.hpp>

#include "ode/MultimodeNLSE.hpp"
#include "ode/InitialCondition.hpp"

#include <nlopt.hpp>

using namespace autodiffeq;

double CalcObjective(const std::vector<double>& x, void* data)
{
  using Complex = complex<double>;
  using clock = std::chrono::high_resolution_clock;

  MultimodeNLSE<Complex>* ode = reinterpret_cast<MultimodeNLSE<Complex>*>(data);

  Array1D<double> Et = {9.0, 8.0}; //nJ (in range [6,30] nJ)
  Array1D<double> t_FWHM = {0.1, 0.2}; //ps (in range [0.05, 0.5] ps)
  Array1D<double> t_center = {x[0], x[1]}; //ps

  Array1D<Complex> sol0;
  ComputeGaussianPulse(Et, t_FWHM, t_center, ode->GetTimeVector(), sol0);

  // for (int i = 0; i < sol0.size(); ++i)
  //   std::cout << i << ": " << "(" << sol0[i].value().real() << ", " << sol0[i].value().imag() << "), " 
  //             << "(" << sol0[i].deriv(0).real() << ", " << sol0[i].deriv(0).imag() << "), " 
  //             << "(" << sol0[i].deriv(1).real() << ", " << sol0[i].deriv(1).imag() << ")" << std::endl;

  double z_start = 0, z_end = 0.1; //[m]
  int nz = 200*20;
  int storage_stride = nz;

  const int order = 4;
  RungeKutta<Complex> solver(*ode, order);
  solver.SetVerbosity(true);

  std::cout << "Solving ODE..." << std::endl;

  auto t0 = clock::now();
  auto sol_hist = solver.Solve(sol0, z_start, z_end, nz, storage_stride);
  auto t1 = clock::now();
  auto time_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count() / 1000.0;
  std::cout << std::fixed << std::setprecision(3);
  std::cout << "Solved in " << time_elapsed << " secs." << std::endl;

  const int num_modes = 2;
  const auto& tvec = ode->GetTimeVector();
  const int num_time_points = tvec.size();
  int step = sol_hist.GetNumSteps() - 1;

  double A_target = 100.0;
  double tc_target = 10.0; //ps

  std::cout << std::scientific << std::setprecision(4);

  double cost = 0.0;
  for (int mode = 0; mode < num_modes; ++mode)
    for (int t = 0; t < num_time_points; ++t)
    {
      // auto u = sol_hist(step, t*num_modes + mode);
      double intensity = abs(sol_hist(step, t*num_modes + mode));

      // std::cout << "u[" << t << ", " << mode << "]: (" << u.value().real() << ", " << u.value().imag() << "), ";
      // if (u.size() > 0)
      //   std::cout << "(" << u.deriv(0).real() << ", " << u.deriv(0).imag() << "), ("
      //             << u.deriv(1).real() << ", " << u.deriv(1).imag() << ")";

      // std::cout << ", I: " << intensity.value().real();
      // if (intensity.size() > 0)
      // {
      // std::cout << ", (" << intensity.deriv(0).real() << ", " << intensity.deriv(0).imag() << "), ("
      //           << intensity.deriv(1).real() << ", " << intensity.deriv(1).imag() << ")";
      // }
      // std::cout << std::endl;
      double target = A_target * exp(-10*(tvec[t] - tc_target)*(tvec[t] - tc_target));
      cost += (intensity - target) * (intensity - target);
    }

  return cost;
}

double CalcObjective(const std::vector<double>& x, std::vector<double>& grad, void* data)
{
  using Complex = complex<double>;
  using ComplexAD = ADVarS<4,complex<double>>;
  using clock = std::chrono::high_resolution_clock;

  MultimodeNLSE<ComplexAD>* ode = reinterpret_cast<MultimodeNLSE<ComplexAD>*>(data);

  std::cout << "x: " << x[0] << ", " << x[1] << ", " << x[2] << ", " << x[3] << std::endl;

  Array1D<double> Et = {9.0, 8.0}; //nJ (in range [6,30] nJ)
  Array1D<ComplexAD> t_FWHM = {ComplexAD(x[0]), 
                               ComplexAD(x[1])}; //ps (in range [0.05, 0.5] ps)
  t_FWHM[0].deriv(0) = 1.0;
  t_FWHM[1].deriv(1) = 1.0;
  Array1D<ComplexAD> t_center = {ComplexAD(x[2]), 
                                 ComplexAD(x[3])}; //ps
  t_center[0].deriv(2) = 1.0;
  t_center[1].deriv(3) = 1.0;

  Array1D<ComplexAD> sol0;
  ComputeGaussianPulse(Et, t_FWHM, t_center, ode->GetTimeVector(), sol0);

  // for (int i = 0; i < sol0.size(); ++i)
  //   std::cout << i << ": " << "(" << sol0[i].value().real() << ", " << sol0[i].value().imag() << "), " 
  //             << "(" << sol0[i].deriv(0).real() << ", " << sol0[i].deriv(0).imag() << "), " 
  //             << "(" << sol0[i].deriv(1).real() << ", " << sol0[i].deriv(1).imag() << ")" << std::endl;


  double z_start = 0, z_end = 0.1; //[m]
  int nz = 200*20;
  int storage_stride = nz;

  const int order = 4;
  RungeKutta<ComplexAD> solver(*ode, order);
  solver.SetVerbosity(true);

  std::cout << "Solving ODE..." << std::endl;

  auto t0 = clock::now();
  auto sol_hist = solver.Solve(sol0, z_start, z_end, nz, storage_stride);
  auto t1 = clock::now();
  auto time_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count() / 1000.0;
  std::cout << std::fixed << std::setprecision(3);
  std::cout << "Solved in " << time_elapsed << " secs." << std::endl;

  const int num_modes = 2;
  const auto& tvec = ode->GetTimeVector();
  const int num_time_points = tvec.size();
  int step = sol_hist.GetNumSteps() - 1;

  double A_target = 100.0;
  double tc_target = 10.0; //ps

  std::cout << std::scientific << std::setprecision(4);

  ComplexAD cost(0);
  for (int mode = 0; mode < num_modes; ++mode)
    for (int t = 0; t < num_time_points; ++t)
    {
      // auto u = sol_hist(step, t*num_modes + mode);
      auto intensity = abs(sol_hist(step, t*num_modes + mode));

      // std::cout << "u[" << t << ", " << mode << "]: (" << u.value().real() << ", " << u.value().imag() << "), ";
      // if (u.size() > 0)
      //   std::cout << "(" << u.deriv(0).real() << ", " << u.deriv(0).imag() << "), ("
      //             << u.deriv(1).real() << ", " << u.deriv(1).imag() << ")";

      // std::cout << ", I: " << intensity.value().real();
      // if (intensity.size() > 0)
      // {
      // std::cout << ", (" << intensity.deriv(0).real() << ", " << intensity.deriv(0).imag() << "), ("
      //           << intensity.deriv(1).real() << ", " << intensity.deriv(1).imag() << ")";
      // }
      // std::cout << std::endl;
      double target = A_target * exp(-10*(tvec[t] - tc_target)*(tvec[t] - tc_target));
      cost += (intensity - target) * (intensity - target);
    }

  std::cout << "cost: " << cost.value().real() << std::endl;
  for (int i = 0; i < x.size(); ++i)
  {
    std::cout << "grad[" << i << "]: " << cost.deriv(i).real() << ", " << cost.deriv(i).imag() << std::endl;
    grad[i] = cost.deriv(i).real();
  }

  return cost.value().real();
}

int main()
{
  using Complex = complex<double>;
  using ComplexAD = ADVarS<4,complex<double>>;
  using clock = std::chrono::high_resolution_clock;

#ifdef ENABLE_OPENMP
  int num_threads = omp_get_max_threads();
#else
  int num_threads = 1;
#endif
  std::cout << "No. of CPU threads available: " << num_threads << std::endl;

  const int num_modes = 2;
  const int num_time_points = 4097;

  Array2D<double> beta_mat_5x8 = 
    {{ 0.00000000e+00, -5.31830434e+03, -5.31830434e+03, -1.06910098e+04, -1.06923559e+04, -1.07426928e+04, -2.16527479e+04, -3.26533894e+04},
     { 0.00000000e+00,  1.19405403e-01,  1.19405403e-01,  2.44294517e-01,  2.43231165e-01,  2.44450336e-01,  5.03940297e-01,  6.85399771e-01},
     {-2.80794698e-02, -2.82196091e-02, -2.82196091e-02, -2.83665602e-02, -2.83665268e-02, -2.83659537e-02, -2.86356131e-02, -2.77985757e-02},
     { 1.51681819e-04,  1.52043264e-04,  1.52043264e-04,  1.52419435e-04,  1.52419402e-04,  1.52414636e-04,  1.52667612e-04,  1.39629075e-04},
     {-4.95686317e-07, -4.97023237e-07, -4.97023237e-07, -4.98371203e-07, -4.98371098e-07, -4.98311743e-07, -4.94029250e-07, -3.32523455e-07}};

  const int max_Ap_tderiv = 4;
  Array2D<double> beta_mat(max_Ap_tderiv+1, num_modes);
  for (int i = 0; i <= max_Ap_tderiv; ++i)
    for (int j = 0; j < num_modes; ++j)
      beta_mat(i,j) = beta_mat_5x8(i,j);

  Array4D<double> Sk(2,2,2,2, {4.9840660e+09, 0.0000000e+00, 0.0000000e+00, 2.5202004e+09,
                               0.0000000e+00, 2.5202004e+09, 2.5202004e+09, 0.0000000e+00,
                               0.0000000e+00, 2.5202004e+09, 2.5202004e+09, 0.0000000e+00,
                               2.5202004e+09, 0.0000000e+00, 0.0000000e+00, 3.7860385e+09});

  double tmin = -40, tmax = 40;
  double n2 = 2.3e-20;
  double omega0 = 1.2153e3;
  bool is_self_steepening = false;
  bool is_nonlinear = true;

  MultimodeNLSE<ComplexAD> ode(num_modes, num_time_points, tmin, tmax, beta_mat,
                               n2, omega0, Sk, is_self_steepening, is_nonlinear);

  //nlopt optimizer object
  int num_params = 4;
  nlopt::opt opt(nlopt::LD_LBFGS, num_params);

  opt.set_min_objective( CalcObjective, reinterpret_cast<void*>(&ode) );

  //stop when every parameter changes by less than the tolerance multiplied by the absolute value of the parameter.
  opt.set_xtol_rel(1e-6);

  //stop when the objective function value changes by less than the tolerance multiplied by the absolute value of the function value
  opt.set_ftol_rel(1e-8);

  //stop when the maximum number of function evaluations is reached
  opt.set_maxeval(20);

  std::vector<double> lower_bounds(num_params), upper_bounds(num_params);
  lower_bounds[0] = 0.05; upper_bounds[0] = 0.5; //ps
  lower_bounds[1] = 0.05; upper_bounds[1] = 0.5; //ps
  lower_bounds[2] = -20.0; upper_bounds[2] = 20.0; //ps
  lower_bounds[3] = -20.0; upper_bounds[3] = 20.0; //ps
  opt.set_lower_bounds(lower_bounds);
  opt.set_upper_bounds(upper_bounds);

  std::vector<double> x = {0.1, 0.1, 9.0, 9.0};

  double cost_opt;
  nlopt::result result = nlopt::result::FAILURE;

  try
  {
    result = opt.optimize(x, cost_opt);
  }
  catch (std::exception &e)
  {
    std::cout << e.what() << std::endl;
    //throwError("NLopt failed!");
  }

  std::cout << "x_opt: " << x[0] << ", " << x[1] << ", " << x[2] << ", " << x[3] << std::endl;

  //Solve ODE with optimal pulse parameters
  MultimodeNLSE<Complex> ode2(num_modes, num_time_points, tmin, tmax, beta_mat,
                              n2, omega0, Sk, is_self_steepening, is_nonlinear);
  Array1D<double> Et = {9.0, 8.0}; //nJ (in range [6,30] nJ)
  Array1D<double> t_FWHM = {x[0], x[1]}; //ps (in range [0.05, 0.5] ps)
  Array1D<double> t_center = {x[2], x[3]}; //ps
  Array1D<Complex> sol0;
  ComputeGaussianPulse(Et, t_FWHM, t_center, ode2.GetTimeVector(), sol0);

  double z_start = 0, z_end = 0.1; //[m]
  int nz = 200*20;
  int storage_stride = 20;

  const int order = 4;
  RungeKutta<Complex> solver(ode2, order);
  solver.SetVerbosity(true);

  std::cout << "Solving ODE..." << std::endl;

  auto t0 = clock::now();
  auto sol_hist = solver.Solve(sol0, z_start, z_end, nz, storage_stride);
  auto t1 = clock::now();
  auto time_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count() / 1000.0;
  std::cout << std::fixed << std::setprecision(3);
  std::cout << "Solved in " << time_elapsed << " secs." << std::endl;

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

  // std::vector<double> grad(x.size());
  // double cost = CalcObjective(x, grad, &ode);

  // MultimodeNLSE<Complex> ode2(num_modes, num_time_points, tmin, tmax, beta_mat,
  //                              n2, omega0, Sk, is_self_steepening, is_nonlinear);
  // double cost0 = CalcObjective(x, &ode2);
  // std::cout << "Cost: " << cost0 << std::endl;

  // double eps = 1e-2;
  // for (int i = 0; i < 2; ++i)
  // {
  //   x[i] += eps;
  //   double costp = CalcObjective(x, &ode2);
  //   x[i] -= 2*eps;
  //   double costm = CalcObjective(x, &ode2);
  //   x[i] += eps;
  //   std::cout << "fd" << i << ": " << (costp - costm) / (2*eps) << std::endl;
  // }

}