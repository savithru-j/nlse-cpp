// nlse++
// C++ library for solving the nonlinear Schrödinger equation
// Licensed under the MIT License (http://opensource.org/licenses/MIT)
// Copyright (c) 2023, Savithru Jayasinghe

#define _USE_MATH_DEFINES //For MSVC
#include <cmath>

#include <autodiffeq/linearalgebra/Array1D.hpp>

namespace autodiffeq
{

template<typename Te, typename Tf, typename Tc, typename Ts>
inline void ComputeGaussianPulse(const Array1D<Te>& Et, const Array1D<Tf>& t_FWHM, const Array1D<Tc>& t_center,
                                 const Array1D<double>& time_vec, Array1D<Ts>& sol) 
{
  const int num_modes = (int) Et.size();
  const int num_time_points = (int) time_vec.size();
  assert(num_modes == (int) t_FWHM.size());
  assert(num_modes == (int) t_center.size());
  sol.resize(num_modes * num_time_points);

  for (int mode = 0; mode < num_modes; ++mode)
  {
    const auto A = sqrt(1665.0*Et(mode) / ((double)num_modes * t_FWHM(mode) * std::sqrt(M_PI)));
    const auto k = -1.665*1.665/(2.0*t_FWHM(mode)*t_FWHM(mode));
    const Tc& tc = t_center(mode);

    #pragma omp parallel for
    for (int j = 0; j < num_time_points; ++j)
      sol(j*num_modes + mode) = A * exp(k*(time_vec(j)-tc)*(time_vec(j)-tc));
  }
}

}