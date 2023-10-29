#include <autodiffeq/numerics/ADVar.hpp>
#include <autodiffeq/numerics/Complex.hpp>
#include <autodiffeq/solver/ForwardEuler.hpp>
#include <autodiffeq/solver/RungeKutta.hpp>
#include <autodiffeq/linearalgebra/Array2D.hpp>
#include <autodiffeq/linearalgebra/Array3D.hpp>
#include <autodiffeq/linearalgebra/Array4D.hpp>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>

#include "ode/MultimodeNLSE.hpp"

#ifdef ENABLE_OPENMP
#include <omp.h>
#endif

using namespace autodiffeq;

int main()
{
  using Complex = complex<double>;
  // using ComplexAD = ADVar<Complex>;
  using clock = std::chrono::high_resolution_clock;

#ifdef ENABLE_OPENMP
  int num_threads = omp_get_max_threads();
#else
  int num_threads = 1;
#endif
  std::cout << "No. of CPU threads available: " << num_threads << std::endl;

  const int num_modes = 2;
  const int num_time_points = 8193; //8192;

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
  bool is_nonlinear = false;

  MultimodeNLSE<Complex> ode(num_modes, num_time_points, tmin, tmax, beta_mat,
                             n2, omega0, Sk, is_self_steepening, is_nonlinear);

  Array1D<double> Et = {9.0, 8.0}; //nJ (in range [6,30] nJ)
  Array1D<double> t_FWHM = {0.1, 0.2}; //ps (in range [0.05, 0.5] ps)
  Array1D<double> t_center = {0.0, 0.0}; //ps
  Array1D<Complex> sol0 = ode.GetInitialSolutionGaussian(Et, t_FWHM, t_center);

  // for (int i = 0; i < num_time_points; ++i)
  //     std::cout << std::abs(sol0(i)) << ", " << std::abs(sol0(num_time_points + i)) << std::endl;

  double z_start = 0, z_end = 7.5; //[m]
  int nz = 15000*20;
  int storage_stride = 100*20;

  // double z_start = 0, z_end = 1.0; //[m]
  // int nz = 2000*20;
  // int storage_stride = 100*20;

  std::cout << "Problem parameters:\n"
            << "  Time range            : [" << tmin << ", " << tmax << "] ps\n"
            << "  Z max                 : " << z_end << " m\n"
            << "  No. of z-steps        : " << nz << "\n"
            << "  Solution storage freq.: " << "Every " << storage_stride << " steps\n" 
            << std::endl;

  const int order = 4;
  RungeKutta<Complex> solver(ode, order);

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
    const int offset = mode*num_time_points;
    for (int i = 0; i < sol_hist.GetNumSteps(); ++i)
    {
      if (i % storage_stride == 0)
      {
      for (int j = 0; j < num_time_points-1; ++j)
        f << abs(sol_hist(i, offset + j)) << ", ";
      f << abs(sol_hist(i, offset + num_time_points-1)) << std::endl;
      }
    }
    f.close();
  }
}