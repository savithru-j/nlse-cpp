#include <autodiffeq/numerics/ADVar.hpp>
#include <autodiffeq/numerics/Complex.hpp>
#include <autodiffeq/solver/ODE.hpp>
#include <autodiffeq/linearalgebra/GPUArray1D.cuh>
#include <autodiffeq/linearalgebra/GPUArray2D.cuh>
#include <iostream>
#include <iomanip>

namespace autodiffeq
{

namespace gpu
{


template<typename T>
__global__
void mmnlseRHS(const int num_modes, const int num_time_pts, const DeviceArray2D<double>& beta_mat, 
               const double dt, const DeviceArray1D<T>& sol, DeviceArray1D<T>& rhs)
{
  const int t = blockIdx.x*blockDim.x + threadIdx.x; //time-point index
  const int p = threadIdx.y; //mode index (blockIdx.y is always 0 since only 1 block is launched in this dimension)
  const int offset = num_modes*t + p;
  int soldim = sol.size();

  const int num_deriv_rows = beta_mat.GetNumRows();
  const int beta_size = num_deriv_rows * num_modes;
  constexpr int stencil_size = 5;

  extern __shared__ double array[];
  double* beta = array; // num_deriv_rows x num_modes doubles
  T* sol_stencil_shared = (T*) &beta[beta_size]; // (num_threads * stencil_size * num_modes) variables of type T

  if (threadIdx.x < num_deriv_rows)
    beta[threadIdx.x*num_modes+p] = beta_mat(threadIdx.x,p);

  int shared_offset = threadIdx.x * stencil_size * num_modes;
  T* sol_stencil = sol_stencil_shared + shared_offset;
  if (t < num_time_pts)
  {
    for (int i = 0; i < stencil_size; ++i)
    {
      int t_offset = t + i - 2;
      if (t_offset < 0)
        t_offset = -t_offset; //Mirror data at left boundary
      if (t_offset >= num_time_pts)
        t_offset = 2*num_time_pts - 2 - t_offset; //Mirror data at right boundary

      sol_stencil[num_modes*i + p] = sol[num_modes*t_offset + p];
    }
  }

  __syncthreads();

  if (offset >= soldim)
    return;

  constexpr double inv6 = 1.0/6.0;
  constexpr double inv24 = 1.0/24.0;
  constexpr T imag(0.0, 1.0);
  const double inv_dt = 1.0/dt;
  const double inv_dt2 = inv_dt*inv_dt;
  const double inv_dt3 = inv_dt2*inv_dt;
  const double inv_dt4 = inv_dt3*inv_dt;

  T sol_im2 = sol_stencil[              p];
  T sol_im1 = sol_stencil[num_modes   + p];
  T sol_i   = sol_stencil[num_modes*2 + p];
  T sol_ip1 = sol_stencil[num_modes*3 + p];
  T sol_ip2 = sol_stencil[num_modes*4 + p];

  //Calculate solution time-derivatives using stencil data
  T sol_tderiv1 = 0.5*(sol_ip1 - sol_im1) * inv_dt;
  T sol_tderiv2 = (sol_ip1 - 2.0*sol_i + sol_im1) * inv_dt2;
  T sol_tderiv3 = (0.5*sol_ip2 - sol_ip1 + sol_im1 - 0.5*sol_im2) * inv_dt3;
  T sol_tderiv4 = (sol_ip2 - 4.0*sol_ip1 + 6.0*sol_i - 4.0*sol_im1 + sol_im2) * inv_dt4;

#if 0
  if (t == 128 && p == 0) {
    printf("sol re: %e, %e, %e, %e, %e\n", sol_im2.real(), sol_im1.real(), sol_i.real(), sol_ip1.real(), sol_ip2.real());
    printf("sol im: %e, %e, %e, %e, %e\n", sol_im2.imag(), sol_im1.imag(), sol_i.imag(), sol_ip1.imag(), sol_ip2.imag());
    printf("tderiv1: %e, %e\n", sol_tderiv1.real(), sol_tderiv1.imag());
    printf("tderiv2: %e, %e\n", sol_tderiv2.real(), sol_tderiv2.imag());
    printf("tderiv3: %e, %e\n", sol_tderiv3.real(), sol_tderiv3.imag());
    printf("tderiv4: %e, %e\n", sol_tderiv4.real(), sol_tderiv4.imag());
    printf("beta: %e, %e, %e, %e, %e\n", beta[p], beta[num_modes+p], beta[2*num_modes+p], beta[3*num_modes+p], beta[4*num_modes+p]);
  }
#endif

  rhs[offset] = imag*(beta[            p] - beta[        0])*sol_i //(beta0p - beta00)
                    -(beta[  num_modes+p] - beta[num_modes])*sol_tderiv1
              - imag* beta[2*num_modes+p]*0.5               *sol_tderiv2
                    + beta[3*num_modes+p]*inv6              *sol_tderiv3
              + imag* beta[4*num_modes+p]*inv24             *sol_tderiv4;
}


}

template<typename T>
class GPUMultimodeNLSE : public ODE<T>
{
public:

  static_assert(std::is_same<T, complex<double>>::value ||
                std::is_same<T, ADVar<complex<double>>>::value, 
                "Template datatype needs to be complex<double> or ADVar<complex<double>>!");

  using ODE<T>::EvalRHS;

  GPUMultimodeNLSE(const int num_modes, const int num_time_points, 
                   const double tmin, const double tmax, const Array2D<double>& beta_mat) :
    GPUMultimodeNLSE(num_modes, num_time_points, tmin, tmax, beta_mat, 0.0, 0.0, {}, false, false) {}

  GPUMultimodeNLSE(const int num_modes, const int num_time_points, 
                   const double tmin, const double tmax, const Array2D<double>& beta_mat,
                   const double n2, const double omega0, const Array4D<double>& Sk, 
                   const bool is_self_steepening, const bool is_nonlinear = true) : 
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

    // const int max_Ap_tderiv = beta_mat_.GetNumRows()-1;
    // sol_tderiv_.resize(max_Ap_tderiv, num_time_points_); //Stores the time-derivatives (e.g. d/dt, d^2/dt^2 ...) of a particular solution mode

    // if (is_nonlinear_)
    // {
    //   kerr_.resize(num_time_points_);
    //   if (is_self_steepening_)
    //     kerr_tderiv_.resize(num_time_points_);
    // }
  }

  int GetSolutionSize() const { return num_modes_*num_time_points_; }

  void EvalRHS(const GPUArray1D<T>& sol, int step, double z, GPUArray1D<T>& rhs)
  {
    constexpr int threads_per_block = 256;
    dim3 thread_dim(threads_per_block, num_modes_, 1);
    dim3 block_dim((num_time_points_ + thread_dim.x-1) / thread_dim.x, 1, 1);
    //std::cout << "thread_dim: " << thread_dim.x << ", " << thread_dim.y << ", " << thread_dim.z << std::endl;
    //std::cout << "block_dim: " << block_dim.x << ", " << block_dim.y << ", " << block_dim.z << std::endl;

    constexpr int stencil_size = 5;
    int shared_mem_bytes = beta_mat_.size()*sizeof(double)
                         + (threads_per_block * stencil_size * num_modes_)*sizeof(T);
    //std::cout << "shared mem bytes: " << shared_mem_bytes << std::endl;

    gpu::mmnlseRHS<<<block_dim, thread_dim, shared_mem_bytes>>>(
      num_modes_, num_time_points_, beta_mat_.GetDeviceArray(), dt_,
      sol.GetDeviceArray(), rhs.GetDeviceArray());
    cudaCheckLastError();

#if 0
    static int iter = 0;
    if (iter == 0)
    {
      auto rhs_cpu = rhs.CopyToHost();
      for (int i = 0; i < num_time_points_; ++i)
      {
        for (int p = 0; p < num_modes_; ++p)
        {
          const auto& v = rhs_cpu(num_modes_*i + p);
          if (v.real() != 0.0 || v.imag() != 0.0)
            std::cout << i << ", " << p << ": " << v.real() << ", " << v.imag() << std::endl;
        }
      }
      exit(0);
    }
    iter++;
#endif
  }

  void ComputeKerrNonlinearity(const int p, const Array1D<T>& sol)
  {
    #pragma omp for
    for (int i = 0; i < num_time_points_; ++i) 
    {
      T sum = 0.0;
      for (int q = 0; q < num_modes_; ++q)
      {
        const auto& Aq = sol(q*num_time_points_ + i);
        for (int r = 0; r < num_modes_; ++r)
        {
          const auto& Ar = sol(r*num_time_points_ + i);
          for (int s = 0; s < num_modes_; ++s)
          {
            const auto& As = sol(s*num_time_points_ + i);
            sum += Sk_(p,q,r,s)*Aq*Ar*conj(As);
          }
        }
      }
      kerr_(i) = sum;
    }
  }

  Array1D<T> GetInitialSolutionGaussian(const Array1D<double>& Et, const Array1D<double>& t_FWHM, const Array1D<double>& t_center) 
  {
    assert(num_modes_ == (int) Et.size());
    assert(num_modes_ == (int) t_FWHM.size());
    assert(num_modes_ == (int) t_center.size());
    Array1D<T> sol(GetSolutionSize());

    for (int mode = 0; mode < num_modes_; ++mode)
    {
      const double A = std::sqrt(1665.0*Et(mode) / ((double)num_modes_ * t_FWHM(mode) * std::sqrt(M_PI)));
      const double k = -1.665*1.665/(2.0*t_FWHM(mode)*t_FWHM(mode));
      const double& tc = t_center(mode);

      for (int j = 0; j < num_time_points_; ++j)
        sol(j*num_modes_ + mode) = A * std::exp(k*(tvec_(j)-tc)*(tvec_(j)-tc));
    }
    return sol;
  }

protected:
  const int num_modes_;
  const int num_time_points_;
  const double tmin_, tmax_, dt_;
  Array1D<double> tvec_;
  GPUArray2D<double> beta_mat_;
  const double n2_; //[m^2 / W]
  const double omega0_; //[rad/ps]
  const Array4D<double>& Sk_; //Kerr nonlinearity tensor
  static constexpr double c_ = 2.99792458e-4; //[m/ps]
  const bool is_self_steepening_ = false;
  const bool is_nonlinear_ = false;
  // Array2D<T> sol_tderiv_;
  Array1D<T> kerr_;
  // Array1D<T> kerr_tderiv_;
};

}