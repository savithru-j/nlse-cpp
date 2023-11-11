// nlse++
// C++ library for solving the nonlinear Schrödinger equation
// Licensed under the MIT License (http://opensource.org/licenses/MIT)
// Copyright (c) 2023, Savithru Jayasinghe

#include "MultimodeNLSE.hpp"

namespace autodiffeq
{

template<typename T>
MultimodeNLSE<T>::MultimodeNLSE(const int num_modes, const int num_time_points, 
                const double tmin, const double tmax, const Array2D<double>& beta_mat) :
    MultimodeNLSE(num_modes, num_time_points, tmin, tmax, beta_mat, 0.0, 0.0, {}, false, false) {}

template<typename T>
MultimodeNLSE<T>::MultimodeNLSE(
  const int num_modes, const int num_time_points, const double tmin, const double tmax, 
  const Array2D<double>& beta_mat, const double n2, const double omega0, const Array4D<double>& Sk, 
  const bool is_self_steepening, const bool is_nonlinear) : 
  num_modes_(num_modes), num_time_points_(num_time_points), tmin_(tmin), tmax_(tmax),
  dt_((tmax_ - tmin_) / (double) (num_time_points_-1)), beta_mat_(beta_mat), n2_(n2),
  omega0_(omega0), Sk_(Sk), is_self_steepening_(is_self_steepening), is_nonlinear_(is_nonlinear)
{
  tvec_.resize(num_time_points_);
  tvec_(0) = tmin_;
  for (int step = 1; step < num_time_points_; ++step)
    tvec_(step) = tvec_(step-1) + dt_;

  assert(beta_mat_.GetNumCols() == num_modes);
  if (is_nonlinear_)
    for (int d = 0; d < 4; ++d)
      assert(Sk.GetDim(d) == num_modes);

  const int max_Ap_tderiv = beta_mat_.GetNumRows()-1;
  sol_tderiv_.resize(max_Ap_tderiv, num_time_points_, T(0)); //Stores the time-derivatives (e.g. d/dt, d^2/dt^2 ...) of a particular solution mode
}

template<typename T>
void 
MultimodeNLSE<T>::EvalRHS(const Array1D<T>& sol, int step, double z, Array1D<T>& rhs)
{
  constexpr complex<double> imag(0.0, 1.0);
  const auto beta00 = beta_mat_(0,0);
  const auto beta10 = beta_mat_(1,0);
  constexpr double inv6 = 1.0/6.0;
  constexpr double inv24 = 1.0/24.0;
  const complex<double> j_n_omega0_invc(0.0, n2_*omega0_/c_);

  for (int p = 0; p < num_modes_; ++p)
  {
    const auto beta0p = beta_mat_(0,p);
    const auto beta1p = beta_mat_(1,p);
    const auto beta2p = beta_mat_(2,p);
    const auto beta3p = beta_mat_(3,p);
    const auto beta4p = beta_mat_(4,p);
    ComputeTimeDerivativesOrder4(p, sol, sol_tderiv_);

    #pragma omp for
    for (int t = 0; t < num_time_points_; ++t) 
    {
      const int offset = num_modes_*t;
      rhs(offset+p) = imag*(beta0p - beta00)*sol(offset+p)
                          -(beta1p - beta10)*sol_tderiv_(0,t) //d/dt
                    - imag* beta2p*0.5      *sol_tderiv_(1,t) //d^2/dt^2
                          + beta3p*inv6     *sol_tderiv_(2,t) //d^3/dt^3
                    + imag* beta4p*inv24    *sol_tderiv_(3,t); //d^4/dt^4
    }

    if (is_nonlinear_)
    {
      #pragma omp for
      for (int t = 0; t < num_time_points_; ++t)
      {
        const int offset = num_modes_*t;
        const T* sol_modes = sol.data() + offset;
        T kerr = T(0);
        for (int q = 0; q < num_modes_; ++q)
        {
          const auto& Aq = sol_modes[q];
          for (int r = 0; r < num_modes_; ++r)
          {
            const auto& Ar = sol_modes[r];
            for (int s = 0; s < num_modes_; ++s)
            {
              const auto& As = sol_modes[s];
              kerr += Sk_(p,q,r,s) * Aq * Ar * conj(As);
            }
          }
        }
        rhs(offset+p) += j_n_omega0_invc*kerr;
      }
    }
  }
}

template<typename T>
void 
MultimodeNLSE<T>::ComputeTimeDerivativesOrder2(const int mode, const Array1D<T>& sol, Array2D<T>& tderiv)
{
  const int max_deriv = tderiv.GetNumRows();
  assert(max_deriv >= 2 && max_deriv <= 4);

  const double inv_dt = 1.0/dt_;
  const double inv_dt2 = inv_dt*inv_dt;
  const double inv_dt3 = inv_dt2*inv_dt;
  const double inv_dt4 = inv_dt3*inv_dt;

  #pragma omp for
  for (int t = 0; t < num_time_points_; ++t)
  {
    //Get solutions at 5 stencil points: mirror data at left and right boundaries
    T sol_im2 = (t >= 2) ? sol(num_modes_*(t-2) + mode) : sol(num_modes_*(2-t) + mode);
    T sol_im1 = (t >= 1) ? sol(num_modes_*(t-1) + mode) : sol(num_modes_*(1-t) + mode);
    const T& sol_i = sol(num_modes_*t + mode);
    T sol_ip1 = (t < num_time_points_-1) ? sol(num_modes_*(t+1) + mode)
                                         : sol(num_modes_*(2*num_time_points_ - t - 3) + mode);
    T sol_ip2 = (t < num_time_points_-2) ? sol(num_modes_*(t+2) + mode)
                                         : sol(num_modes_*(2*num_time_points_ - t - 4) + mode);

    //Calculate solution time-derivatives using stencil data
    tderiv(0,t) = 0.5*(sol_ip1 - sol_im1) * inv_dt; //d/dt
    tderiv(1,t) = (sol_ip1 - 2.0*sol_i + sol_im1) * inv_dt2;
    tderiv(2,t) = (0.5*(sol_ip2 - sol_im2) - sol_ip1 + sol_im1) * inv_dt3;
    tderiv(3,t) = (sol_ip2 - 4.0*(sol_ip1 + sol_im1) + 6.0*sol_i + sol_im2) * inv_dt4;
  }
}

template<typename T>
void 
MultimodeNLSE<T>::ComputeTimeDerivativesOrder4(const int mode, const Array1D<T>& sol, Array2D<T>& tderiv)
{
  const int max_deriv = tderiv.GetNumRows();
  assert(max_deriv >= 2 && max_deriv <= 4);

  const double inv_12dt = 1.0/(12.0*dt_);
  const double inv_12dt2 = inv_12dt / dt_;
  const double inv_8dt3 = 1.0 / (8.0*dt_*dt_*dt_);
  const double inv_6dt4 = 1.0 / (6.0*dt_*dt_*dt_*dt_);

  #pragma omp for
  for (int t = 0; t < num_time_points_; ++t)
  {
    //Get solutions at 7 stencil points: mirror data at left and right boundaries
    T sol_im3 = (t >= 3) ? sol(num_modes_*(t-3) + mode) : sol(num_modes_*(3-t) + mode);
    T sol_im2 = (t >= 2) ? sol(num_modes_*(t-2) + mode) : sol(num_modes_*(2-t) + mode);
    T sol_im1 = (t >= 1) ? sol(num_modes_*(t-1) + mode) : sol(num_modes_*(1-t) + mode);
    const T& sol_i = sol(num_modes_*t + mode);
    T sol_ip1 = (t < num_time_points_-1) ? sol(num_modes_*(t+1) + mode)
                                         : sol(num_modes_*(2*num_time_points_ - t - 3) + mode);
    T sol_ip2 = (t < num_time_points_-2) ? sol(num_modes_*(t+2) + mode)
                                         : sol(num_modes_*(2*num_time_points_ - t - 4) + mode);
    T sol_ip3 = (t < num_time_points_-3) ? sol(num_modes_*(t+3) + mode)
                                         : sol(num_modes_*(2*num_time_points_ - t - 5) + mode);

    //Calculate solution time-derivatives using stencil data
    tderiv(0,t) = (sol_im2 - 8.0*(sol_im1 - sol_ip1) - sol_ip2) * inv_12dt; //d/dt
    tderiv(1,t) = (-sol_im2 + 16.0*(sol_im1 + sol_ip1) - 30.0*sol_i - sol_ip2) * inv_12dt2;
    tderiv(2,t) = (sol_im3 - 8.0*(sol_im2 - sol_ip2) + 13.0*(sol_im1 - sol_ip1) - sol_ip3) * inv_8dt3;
    tderiv(3,t) = (-(sol_im3 + sol_ip3) + 12.0*(sol_im2 + sol_ip2) - 39.0*(sol_im1 + sol_ip1) + 56.0*sol_i) * inv_6dt4;
  }
}

//Explicit instantiations
template class MultimodeNLSE<complex<double>>;
template class MultimodeNLSE<ADVar<complex<double>>>;
}