// nlse++
// C++ library for solving the nonlinear Schrödinger equation
// Licensed under the MIT License (http://opensource.org/licenses/MIT)
// Copyright (c) 2023, Savithru Jayasinghe

#include <autodiffeq/numerics/ADVar.hpp>
#include <autodiffeq/numerics/Complex.hpp>
#include <autodiffeq/solver/ODE.hpp>
#include <autodiffeq/linearalgebra/GPUArray1D.cuh>
#include <autodiffeq/linearalgebra/GPUArray2D.cuh>
#include <autodiffeq/linearalgebra/GPUArray4D.cuh>
#include <iostream>
#include <iomanip>

namespace autodiffeq
{

namespace gpu
{

template<typename T>
__global__
void AddDispersionStencil5SharedMem(const int num_modes, const int num_time_pts,
                                    const DeviceArray2D<double>& beta_mat, const double dt,
                                    const DeviceArray1D<T>& sol, DeviceArray1D<T>& rhs)
{
  const int t = blockIdx.x*blockDim.x + threadIdx.x; //time-point index
  const int p = threadIdx.y; //mode index (blockIdx.y is always 0 since only 1 block is launched in this dimension)
  const int offset = num_modes*t + p;

  const int num_deriv_rows = beta_mat.GetNumRows();
  int beta_size = num_deriv_rows * num_modes;
  beta_size += (beta_size % 2); //Round up to multiple of two to avoid misaligned memory error
  constexpr int stencil_size = 5;
  constexpr int stencil_mid = stencil_size / 2;

  extern __shared__ double array[];
  double* beta = array; // num_deriv_rows x num_modes doubles
  T* sol_stencil_shared = (T*) &beta[beta_size]; // (num_threads * stencil_size * num_modes) variables of type T

  if (threadIdx.x < num_deriv_rows && p < num_modes)
    beta[threadIdx.x*num_modes+p] = beta_mat(threadIdx.x,p);

  int shared_offset = threadIdx.x * stencil_size * num_modes;
  T* sol_stencil = sol_stencil_shared + shared_offset;
  if (t < num_time_pts && p < num_modes)
  {
    for (int i = 0; i < stencil_size; ++i)
    {
      int t_offset = t + i - stencil_mid;
      if (t_offset < 0)
        t_offset = -t_offset; //Mirror data at left boundary
      if (t_offset >= num_time_pts)
        t_offset = 2*num_time_pts - 2 - t_offset; //Mirror data at right boundary

      sol_stencil[num_modes*i + p] = sol[num_modes*t_offset + p];
    }
  }

  __syncthreads();

  if (t >= num_time_pts || p >= num_modes)
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
  T sol_tderiv3 = (0.5*(sol_ip2 - sol_im2) - sol_ip1 + sol_im1) * inv_dt3;
  T sol_tderiv4 = (sol_ip2 - 4.0*(sol_ip1 + sol_im1) + 6.0*sol_i + sol_im2) * inv_dt4;

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

  T rhs_val = imag*(beta[            p] - beta[        0])*sol_i //(beta0p - beta00)
                  -(beta[  num_modes+p] - beta[num_modes])*sol_tderiv1
            - imag* beta[2*num_modes+p]*0.5               *sol_tderiv2
                  + beta[3*num_modes+p]*inv6              *sol_tderiv3
            + imag* beta[4*num_modes+p]*inv24             *sol_tderiv4;
  rhs[offset] = rhs_val;
}

template<typename T>
__global__
void AddDispersionStencil7SharedMem(const int num_modes, const int num_time_pts,
                                    const DeviceArray2D<double>& beta_mat, const double dt,
                                    const DeviceArray1D<T>& sol, DeviceArray1D<T>& rhs)
{
  const int t = blockIdx.x*blockDim.x + threadIdx.x; //time-point index
  const int p = threadIdx.y; //mode index (blockIdx.y is always 0 since only 1 block is launched in this dimension)
  const int offset = num_modes*t + p;

  const int num_deriv_rows = beta_mat.GetNumRows();
  int beta_size = num_deriv_rows * num_modes;
  beta_size += (beta_size % 2); //Round up to multiple of two to avoid misaligned memory error
  constexpr int stencil_size = 7;
  constexpr int stencil_mid = stencil_size / 2;

  extern __shared__ double array[];
  double* beta = array; // num_deriv_rows x num_modes doubles
  T* sol_stencil_shared = (T*) &beta[beta_size]; // (num_threads * stencil_size * num_modes) variables of type T

  if (threadIdx.x < num_deriv_rows && p < num_modes)
    beta[threadIdx.x*num_modes+p] = beta_mat(threadIdx.x,p);

  int shared_offset = threadIdx.x * stencil_size * num_modes;
  T* sol_stencil = sol_stencil_shared + shared_offset;
  if (t < num_time_pts && p < num_modes)
  {
    for (int i = 0; i < stencil_size; ++i)
    {
      int t_offset = t + i - stencil_mid;
      if (t_offset < 0)
        t_offset = -t_offset; //Mirror data at left boundary
      if (t_offset >= num_time_pts)
        t_offset = 2*num_time_pts - 2 - t_offset; //Mirror data at right boundary

      sol_stencil[num_modes*i + p] = sol[num_modes*t_offset + p];
    }
  }

  __syncthreads();

  if (t >= num_time_pts || p >= num_modes)
    return;

  constexpr double inv6 = 1.0/6.0;
  constexpr double inv24 = 1.0/24.0;
  constexpr T imag(0.0, 1.0);
  const double inv_12dt = 1.0/(12.0*dt);
  const double inv_12dt2 = inv_12dt / dt;
  const double inv_8dt3 = 1.0 / (8.0*dt*dt*dt);
  const double inv_6dt4 = 1.0 / (6.0*dt*dt*dt*dt);

  T sol_im3 = sol_stencil[              p];
  T sol_im2 = sol_stencil[num_modes   + p];
  T sol_im1 = sol_stencil[num_modes*2 + p];
  T sol_i   = sol_stencil[num_modes*3 + p];
  T sol_ip1 = sol_stencil[num_modes*4 + p];
  T sol_ip2 = sol_stencil[num_modes*5 + p];
  T sol_ip3 = sol_stencil[num_modes*6 + p];

  //Calculate solution time-derivatives using stencil data
  T sol_tderiv1 = (sol_im2 - 8.0*(sol_im1 - sol_ip1) - sol_ip2) * inv_12dt;
  T sol_tderiv2 = (-sol_im2 + 16.0*(sol_im1 + sol_ip1) - 30.0*sol_i - sol_ip2) * inv_12dt2;
  T sol_tderiv3 = (sol_im3 - 8.0*(sol_im2 - sol_ip2) + 13.0*(sol_im1 - sol_ip1) - sol_ip3) * inv_8dt3;
  T sol_tderiv4 = (-(sol_im3 + sol_ip3) + 12.0*(sol_im2 + sol_ip2) - 39.0*(sol_im1 + sol_ip1) + 56.0*sol_i) * inv_6dt4;

  T rhs_val = imag*(beta[            p] - beta[        0])*sol_i //(beta0p - beta00)
                  -(beta[  num_modes+p] - beta[num_modes])*sol_tderiv1
            - imag* beta[2*num_modes+p]*0.5               *sol_tderiv2
                  + beta[3*num_modes+p]*inv6              *sol_tderiv3
            + imag* beta[4*num_modes+p]*inv24             *sol_tderiv4;
  rhs[offset] = rhs_val;
}

template<typename T>
__global__
void AddDispersionStencil5(const int num_modes, const int num_time_pts,
                            const DeviceArray2D<double>& beta_mat, const double dt,
                            const DeviceArray1D<T>& sol, DeviceArray1D<T>& rhs)
{
  const int t = blockIdx.x*blockDim.x + threadIdx.x; //time-point index
  const int p = threadIdx.y; //mode index (blockIdx.y is always 0 since only 1 block is launched in this dimension)
  const int offset = num_modes*t + p;
  constexpr int stencil_size = 5;
  constexpr int stencil_mid = stencil_size / 2;

  if (t >= num_time_pts || p >= num_modes)
    return;

  constexpr double inv6 = 1.0/6.0;
  constexpr double inv24 = 1.0/24.0;
  constexpr T imag(0.0, 1.0);
  const double inv_dt = 1.0/dt;
  const double inv_dt2 = inv_dt*inv_dt;
  const double inv_dt3 = inv_dt2*inv_dt;
  const double inv_dt4 = inv_dt3*inv_dt;

  int t_offset[stencil_size];
  for (int i = 0; i < stencil_size; ++i)
  {
    t_offset[i] = t + i - stencil_mid;
    if (t_offset[i] < 0)
      t_offset[i] = -t_offset[i]; //Mirror data at left boundary
    if (t_offset[i] >= num_time_pts)
      t_offset[i] = 2*num_time_pts - 2 - t_offset[i]; //Mirror data at right boundary
  }

  T sol_im2 = sol[num_modes*t_offset[0] + p];
  T sol_im1 = sol[num_modes*t_offset[1] + p];
  T sol_i   = sol[num_modes*t_offset[2] + p];
  T sol_ip1 = sol[num_modes*t_offset[3] + p];
  T sol_ip2 = sol[num_modes*t_offset[4] + p];

  //Calculate solution time-derivatives using stencil data
  T sol_tderiv1 = 0.5*(sol_ip1 - sol_im1) * inv_dt;
  T sol_tderiv2 = (sol_ip1 - 2.0*sol_i + sol_im1) * inv_dt2;
  T sol_tderiv3 = (0.5*(sol_ip2 - sol_im2) - sol_ip1 + sol_im1) * inv_dt3;
  T sol_tderiv4 = (sol_ip2 - 4.0*(sol_ip1 + sol_im1) + 6.0*sol_i + sol_im2) * inv_dt4;

  T rhs_val = imag*(beta_mat(0,p) - beta_mat(0,0))*sol_i //(beta0p - beta00)
                  -(beta_mat(1,p) - beta_mat(1,0))*sol_tderiv1
            - imag* beta_mat(2,p)*0.5             *sol_tderiv2
                  + beta_mat(3,p)*inv6            *sol_tderiv3
            + imag* beta_mat(4,p)*inv24           *sol_tderiv4;
  rhs[offset] = rhs_val;
}

template<typename T>
__global__
void AddDispersionStencil7(const int num_modes, const int num_time_pts,
                            const DeviceArray2D<double>& beta_mat, const double dt,
                            const DeviceArray1D<T>& sol, DeviceArray1D<T>& rhs)
{
  const int t = blockIdx.x*blockDim.x + threadIdx.x; //time-point index
  const int p = threadIdx.y; //mode index (blockIdx.y is always 0 since only 1 block is launched in this dimension)
  const int offset = num_modes*t + p;
  constexpr int stencil_size = 7;
  constexpr int stencil_mid = stencil_size / 2;

  if (t >= num_time_pts || p >= num_modes)
    return;

  constexpr double inv6 = 1.0/6.0;
  constexpr double inv24 = 1.0/24.0;
  constexpr T imag(0.0, 1.0);
  const double inv_12dt = 1.0/(12.0*dt);
  const double inv_12dt2 = inv_12dt / dt;
  const double inv_8dt3 = 1.0 / (8.0*dt*dt*dt);
  const double inv_6dt4 = 1.0 / (6.0*dt*dt*dt*dt);

  int t_offset[stencil_size];
  for (int i = 0; i < stencil_size; ++i)
  {
    t_offset[i] = t + i - stencil_mid;
    if (t_offset[i] < 0)
      t_offset[i] = -t_offset[i]; //Mirror data at left boundary
    if (t_offset[i] >= num_time_pts)
      t_offset[i] = 2*num_time_pts - 2 - t_offset[i]; //Mirror data at right boundary
  }

  T sol_im3 = sol[num_modes*t_offset[0] + p];
  T sol_im2 = sol[num_modes*t_offset[1] + p];
  T sol_im1 = sol[num_modes*t_offset[2] + p];
  T sol_i   = sol[num_modes*t_offset[3] + p];
  T sol_ip1 = sol[num_modes*t_offset[4] + p];
  T sol_ip2 = sol[num_modes*t_offset[5] + p];
  T sol_ip3 = sol[num_modes*t_offset[6] + p];

  //Calculate solution time-derivatives using stencil data
  T sol_tderiv1 = (sol_im2 - 8.0*(sol_im1 - sol_ip1) - sol_ip2) * inv_12dt;
  T sol_tderiv2 = (-sol_im2 + 16.0*(sol_im1 + sol_ip1) - 30.0*sol_i - sol_ip2) * inv_12dt2;
  T sol_tderiv3 = (sol_im3 - 8.0*(sol_im2 - sol_ip2) + 13.0*(sol_im1 - sol_ip1) - sol_ip3) * inv_8dt3;
  T sol_tderiv4 = (-(sol_im3 + sol_ip3) + 12.0*(sol_im2 + sol_ip2) - 39.0*(sol_im1 + sol_ip1) + 56.0*sol_i) * inv_6dt4;

  T rhs_val = imag*(beta_mat(0,p) - beta_mat(0,0))*sol_i //(beta0p - beta00)
                  -(beta_mat(1,p) - beta_mat(1,0))*sol_tderiv1
            - imag* beta_mat(2,p)*0.5             *sol_tderiv2
                  + beta_mat(3,p)*inv6            *sol_tderiv3
            + imag* beta_mat(4,p)*inv24           *sol_tderiv4;
  rhs[offset] = rhs_val;
}

template<typename T, bool use_shared_mem_for_Sk = true>
__global__
void AddKerrNonlinearity(const int num_modes, const int num_time_pts,
                         const DeviceArray4D<double>& Sk_tensor, const DeviceArray1D<T>& sol, 
                         complex<double> j_n_omega0_invc, DeviceArray1D<T>& rhs)
{
  const int t = blockIdx.x*blockDim.x + threadIdx.x; //time-point index
  const int p = threadIdx.y; //mode index (blockIdx.y is always 0 since only 1 block is launched in this dimension)
  const int offset = num_modes*t + p;
  int soldim = sol.size();
  const int sol_shared_size = blockDim.x * num_modes;

  extern __shared__ double array[];
  T* sol_shared = (T*) array; // (num_threads * num_modes) variables of type T
  if (t < num_time_pts && p < num_modes)
    sol_shared[threadIdx.x*num_modes+p] = sol[offset];

  double* Sk = nullptr;
  if (use_shared_mem_for_Sk)
  {
    Sk = (double*) &sol_shared[sol_shared_size]; // (num_modes^4) doubles
    const int num_modes_sq = num_modes * num_modes;
    const int num_modes_cu = num_modes * num_modes * num_modes;
    if (threadIdx.x < num_modes_cu && p < num_modes)
    {
      // threadIdx.x = N*(N*i + j) + k = N*N*i + N*j + k
      const int i = threadIdx.x / num_modes_sq;
      const int tmp = threadIdx.x - i*num_modes_sq;
      const int j = tmp / num_modes;
      const int k = tmp % num_modes;
      Sk[threadIdx.x*num_modes+p] = Sk_tensor(i,j,k,p);
    }
  }

  __syncthreads();

  if (t >= num_time_pts || p >= num_modes)
    return;

  T* sol_modes = sol_shared + (threadIdx.x*num_modes);
  T kerr = 0.0;
  if (use_shared_mem_for_Sk)
  {
    const int num_modes_sq = num_modes * num_modes;
    const int num_modes_cu = num_modes * num_modes * num_modes;
    for (int q = 0; q < num_modes; ++q)
    {
      T Aq = sol_modes[q];
      for (int r = 0; r < num_modes; ++r)
      {
        T Ar = sol_modes[r];
        for (int s = 0; s < num_modes; ++s)
        {
          T As = sol_modes[s];
          int Sk_ind = num_modes_cu*p + num_modes_sq*q + num_modes*r + s; //linear index for Sk(p,q,r,s)
          kerr += Sk[Sk_ind]*Aq*Ar*conj(As);
        }
      }
    }
  }
  else
  {
    for (int q = 0; q < num_modes; ++q)
    {
      T Aq = sol_modes[q];
      for (int r = 0; r < num_modes; ++r)
      {
        T Ar = sol_modes[r];
        for (int s = 0; s < num_modes; ++s)
        {
          T As = sol_modes[s];
          kerr += Sk_tensor(p,q,r,s)*Aq*Ar*conj(As);
        }
      }
    }
  }

  rhs[offset] += j_n_omega0_invc*kerr;
}


} //gpu namespace

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
    omega0_(omega0), Sk_(Sk), is_self_steepening_(is_self_steepening), is_nonlinear_(is_nonlinear),
    mode_threads_per_block_(GetNextPowerOfTwo(num_modes_))
  {
    std::cout << std::string(60, '-') << std::endl;
    const int dev_id = 0;
    cudaDeviceGetAttribute(&device_shared_mem_bytes_, cudaDevAttrMaxSharedMemoryPerBlock, dev_id);
    std::cout << "GPU shared memory per block: " << device_shared_mem_bytes_ << " bytes\n";
    std::cout << "GPU block dimensions: : " 
              << time_threads_per_block_ << " x " << mode_threads_per_block_ << "\n";

    tvec_.resize(num_time_points_);
    tvec_(0) = tmin_;
    for (int step = 1; step < num_time_points_; ++step)
      tvec_(step) = tvec_(step-1) + dt_;

    assert(beta_mat_.GetNumCols() == num_modes);
    if (is_nonlinear_)
    {
      for (int d = 0; d < 4; ++d)
        assert(Sk.GetDim(d) == num_modes);

      shared_mem_bytes_Kerr_ = (time_threads_per_block_ * num_modes_)*sizeof(T);
      if (shared_mem_bytes_Kerr_ >= device_shared_mem_bytes_)
      {
        std::cout << "Insufficient shared memory on device to launch Kerr nonlinearity kernel.\n";
        exit(1);
      }
      int num_bytes_Sk = num_modes_*num_modes_*num_modes_*num_modes_*sizeof(double);
      if (shared_mem_bytes_Kerr_ + num_bytes_Sk < device_shared_mem_bytes_)
      {
        shared_mem_bytes_Kerr_ += num_bytes_Sk;
        is_Sk_tensor_in_shared_mem_ = true;
      }
      else
        is_Sk_tensor_in_shared_mem_ = false;

      std::cout << "Shared memory required for Kerr nonlinearity kernel: " 
                << shared_mem_bytes_Kerr_ << " bytes\n";
      if (is_Sk_tensor_in_shared_mem_)
        std::cout << " - Sk tensor data stored in shared memory\n";
      else
        std::cout << " - Sk tensor data stored in global memory\n";
    }

    //Note: beta mat size should be a multiple of two to avoid misaligned memory errors in kernel
    int beta_mat_size_even = beta_mat_.size() + (beta_mat_.size() % 2);
    shared_mem_bytes_dispersion_ = beta_mat_size_even*sizeof(double)
              + (time_threads_per_block_ * dispersion_stencil_size_ * num_modes_)*sizeof(T);
    std::cout << "Shared memory required for dispersion kernel: " 
                << shared_mem_bytes_dispersion_ << " bytes\n";
    const bool force_global_mem_dispersion = false;
    if (shared_mem_bytes_dispersion_ < device_shared_mem_bytes_ && !force_global_mem_dispersion) 
    {
      use_dispersion_shared_mem_ = true;
      std::cout << " - Using shared memory version of dispersion kernel\n";
    }
    else
    {
      use_dispersion_shared_mem_ = false;
      std::cout << " - Using global memory version of dispersion kernel\n";
    }

    std::cout << std::string(60, '-') << std::endl;
  }

  int GetSolutionSize() const { return num_modes_*num_time_points_; }
  const Array1D<double>& GetTimeVector() const { return tvec_; }

  void EvalRHS(const GPUArray1D<T>& sol, int step, double z, GPUArray1D<T>& rhs)
  {
    dim3 block_dim(time_threads_per_block_, mode_threads_per_block_, 1);
    dim3 grid_dim((num_time_points_ + block_dim.x-1) / block_dim.x, 1, 1);
    //std::cout << "block_dim: " << block_dim.x << ", " << block_dim.y << ", " << block_dim.z << std::endl;
    //std::cout << "grid_dim: " << grid_dim.x << ", " << grid_dim.y << ", " << grid_dim.z << std::endl;

    if (dispersion_stencil_size_ == 5)
    {
      //Compute dispersion term using second-order time derivatives
      if (use_dispersion_shared_mem_)
      {
        gpu::AddDispersionStencil5SharedMem<<<grid_dim, block_dim, shared_mem_bytes_dispersion_>>>(
          num_modes_, num_time_points_, beta_mat_.GetDeviceArray(), dt_, 
          sol.GetDeviceArray(), rhs.GetDeviceArray());
      }
      else
      {
        gpu::AddDispersionStencil5<<<grid_dim, block_dim>>>(
          num_modes_, num_time_points_, beta_mat_.GetDeviceArray(), dt_, 
          sol.GetDeviceArray(), rhs.GetDeviceArray());
      }
      cudaCheckLastError();
    }
    else if (dispersion_stencil_size_ == 7)
    {
      //Compute dispersion term using fourth-order time derivatives
      if (use_dispersion_shared_mem_)
      {
        gpu::AddDispersionStencil7SharedMem<<<grid_dim, block_dim, shared_mem_bytes_dispersion_>>>(
          num_modes_, num_time_points_, beta_mat_.GetDeviceArray(), dt_, 
          sol.GetDeviceArray(), rhs.GetDeviceArray());
      }
      else
      {
        gpu::AddDispersionStencil7<<<grid_dim, block_dim>>>(
          num_modes_, num_time_points_, beta_mat_.GetDeviceArray(), dt_, 
          sol.GetDeviceArray(), rhs.GetDeviceArray());
      }
      cudaCheckLastError();
    }

    if (is_nonlinear_)
    {
      const complex<double> j_n_omega0_invc(0.0, n2_*omega0_/c_);
      if (is_Sk_tensor_in_shared_mem_)
      {
        gpu::AddKerrNonlinearity<T, true><<<grid_dim, block_dim, shared_mem_bytes_Kerr_>>>(
          num_modes_, num_time_points_, Sk_.GetDeviceArray(), sol.GetDeviceArray(), 
          j_n_omega0_invc, rhs.GetDeviceArray());
      }
      else
      {
        gpu::AddKerrNonlinearity<T, false><<<grid_dim, block_dim, shared_mem_bytes_Kerr_>>>(
          num_modes_, num_time_points_, Sk_.GetDeviceArray(), sol.GetDeviceArray(), 
          j_n_omega0_invc, rhs.GetDeviceArray());
      }
    }
  }

protected:

  int GetNextPowerOfTwo(int number) const {
    int next_power = 1;
    while (next_power < number) {
      next_power *= 2;
    }
    return next_power;
  }

  const int num_modes_;
  const int num_time_points_;
  const double tmin_, tmax_, dt_;
  Array1D<double> tvec_;
  GPUArray2D<double> beta_mat_;
  const double n2_; //[m^2 / W]
  const double omega0_; //[rad/ps]
  GPUArray4D<double> Sk_; //Kerr nonlinearity tensor
  static constexpr double c_ = 2.99792458e-4; //[m/ps]
  const bool is_self_steepening_ = false;
  const bool is_nonlinear_ = false;

  static constexpr int time_threads_per_block_ = 128;
  const int mode_threads_per_block_ = 0;
  static constexpr int dispersion_stencil_size_ = 7;
  int device_shared_mem_bytes_ = 48000;
  int shared_mem_bytes_dispersion_ = 0;
  int shared_mem_bytes_Kerr_ = 0;
  bool use_dispersion_shared_mem_ = true;
  bool is_Sk_tensor_in_shared_mem_ = true;

};

}