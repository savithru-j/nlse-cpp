// nlse++
// C++ library for solving the nonlinear Schrödinger equation
// Licensed under the MIT License (http://opensource.org/licenses/MIT)
// Copyright (c) 2023, Savithru Jayasinghe

#include <iostream>
#include <iomanip>
#include <autodiffeq/numerics/ADVar.hpp>
#include <autodiffeq/numerics/Complex.hpp>
#include <autodiffeq/solver/ODE.hpp>
#include <autodiffeq/linearalgebra/Array1D.hpp>
#include <autodiffeq/linearalgebra/Array2D.hpp>
#include <autodiffeq/linearalgebra/Array4D.hpp>

namespace autodiffeq
{

template<typename T>
class MultimodeNLSE : public ODE<T>
{
public:

  // static_assert(std::is_same<T, complex<double>>::value ||
  //               std::is_same<T, ADVar<complex<double>>>::value ||
  //               std::is_same<T, ADVarS<1,complex<double>>>::value, 
  //               "Template datatype needs to be complex<double> or ADVar<complex<double>>!");
  using ODE<T>::EvalRHS;

  MultimodeNLSE(const int num_modes, const int num_time_points, 
                const double tmin, const double tmax, const Array2D<double>& beta_mat);

  MultimodeNLSE(const int num_modes, const int num_time_points, 
                const double tmin, const double tmax, const Array2D<double>& beta_mat,
                const double n2, const double omega0, const Array4D<double>& Sk, 
                const bool is_self_steepening, const bool is_nonlinear = true);
  
  inline int GetSolutionSize() const { return num_modes_*num_time_points_; }
  const Array1D<double>& GetTimeVector() const { return tvec_; }

  void EvalRHS(const Array1D<T>& sol, int step, double z, Array1D<T>& rhs) override;

  void ComputeTimeDerivativesOrder2(const int mode, const Array1D<T>& sol, Array2D<T>& tderiv);
  void ComputeTimeDerivativesOrder4(const int mode, const Array1D<T>& sol, Array2D<T>& tderiv);

protected:
  const int num_modes_;
  const int num_time_points_;
  const double tmin_, tmax_, dt_;
  Array1D<double> tvec_;
  const Array2D<double>& beta_mat_;
  const double n2_; //[m^2 / W]
  const double omega0_; //[rad/ps]
  const Array4D<double>& Sk_; //Kerr nonlinearity tensor
  static constexpr double c_ = 2.99792458e-4; //[m/ps]
  const bool is_self_steepening_ = false;
  const bool is_nonlinear_ = false;
  Array2D<T> sol_tderiv_;
};

}