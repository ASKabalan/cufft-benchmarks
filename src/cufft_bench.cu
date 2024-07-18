#include "perfostep.hpp"
#include <argparse.hpp>
#include <assert.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>
#include <exception>
#include <iostream>
#include <random>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#define CHECK_CUFFT_EXIT(call)                                                 \
  do {                                                                         \
    cufftResult_t err = call;                                                  \
    if (CUFFT_SUCCESS != err) {                                                \
      fprintf(stderr, "%s:%d CUFFT error. (error code %d)\n", __FILE__,        \
              __LINE__, err);                                                  \
      throw std::exception();                                                  \
    }                                                                          \
  } while (false)

void checkAccuracyHalf(const __half *h_idata, const __half *h_odata,
                       int signal_size, int batch_size, double tolerance) {
  float max_error = 0.0;
  for (int i = 0; i < signal_size * batch_size; ++i) {
    float original = __half2float(h_idata[i]);
    float inverse_transformed = __half2float(h_odata[i]) / signal_size;
    float error = fabs(original - inverse_transformed);
    if (error > max_error) {
      max_error = error;
    }
  }
  std::cout << "For half precision, max error: " << max_error << std::endl;
  assert(max_error < tolerance);
}

void checkAccuracyFloat(const float *h_idata, const float *h_odata,
                        int signal_size, int batch_size, double tolerance) {
  float max_error = 0.0;
  for (int i = 0; i < signal_size * batch_size; ++i) {
    float original = h_idata[i];
    float inverse_transformed = h_odata[i] / signal_size;
    float error = fabsf(original - inverse_transformed);
    if (error > max_error) {
      max_error = error;
    }
  }
  std::cout << "For float precision, max error: " << max_error << std::endl;
  assert(max_error < tolerance);
}

void checkAccuracyDouble(const double *h_idata, const double *h_odata,
                         int signal_size, int batch_size, double tolerance) {
  double max_error = 0.0;
  for (int i = 0; i < signal_size * batch_size; ++i) {
    double original = h_idata[i];
    double inverse_transformed = h_odata[i] / signal_size;
    double error = fabs(original - inverse_transformed);
    if (error > max_error) {
      max_error = error;
    }
  }
  std::cout << "For double precision, max error: " << max_error << std::endl;
  assert(max_error < tolerance);
}

void batchedFFT_IFFT_Float32(int signal_size, int batch_size, int iterations,
                             Perfostep &perf, const std::string &output_file) {
  cufftReal *d_idata, *d_odata;
  cufftComplex *d_fftdata;
  cufftHandle plan_fft, plan_ifft;
  cudaMalloc(&d_idata, sizeof(cufftReal) * signal_size * batch_size);
  cudaMalloc(&d_fftdata,
             sizeof(cufftComplex) * (signal_size / 2 + 1) * batch_size);
  cudaMalloc(&d_odata, sizeof(cufftReal) * signal_size * batch_size);

  // Data initialization with random data
  std::default_random_engine generator;
  std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
  std::vector<cufftReal> h_idata(signal_size * batch_size);
  for (int i = 0; i < signal_size * batch_size; ++i) {
    h_idata[i] = distribution(generator);
  }
  cudaMemcpy(d_idata, h_idata.data(),
             sizeof(cufftReal) * signal_size * batch_size,
             cudaMemcpyHostToDevice);

  // Create FFT and IFFT plans
  CHECK_CUFFT_EXIT(cufftCreate(&plan_fft));
  CHECK_CUFFT_EXIT(cufftCreate(&plan_ifft));

  // Plan creation benchmark
  size_t workSize;
  long long int signal_size_ll = signal_size;
  ColumnNames cols = {{"Precision", "Float32"},
                      {"Signal Size", std::to_string(signal_size)},
                      {"Batch Size", std::to_string(batch_size)}};
  perf.Start("Creation_FFT", cols);
  CHECK_CUFFT_EXIT(cufftXtMakePlanMany(plan_fft, 1, &signal_size_ll, NULL, 1,
                                       signal_size_ll, CUDA_R_32F, NULL, 1,
                                       signal_size_ll / 2 + 1, CUDA_C_32F,
                                       batch_size, &workSize, CUDA_C_32F));
  perf.Stop();

  perf.Start("Creation_IFFT", cols);
  CHECK_CUFFT_EXIT(cufftXtMakePlanMany(plan_ifft, 1, &signal_size_ll, NULL, 1,
                                       signal_size_ll / 2 + 1, CUDA_C_32F, NULL,
                                       1, signal_size_ll, CUDA_R_32F,
                                       batch_size, &workSize, CUDA_C_32F));
  perf.Stop();

  // FFT and IFFT execution benchmark
  for (int i = 0; i < iterations; ++i) {
    perf.Start("FFT-Iteration " + std::to_string(i + 1), cols);
    CHECK_CUFFT_EXIT(cufftXtExec(plan_fft, d_idata, d_fftdata, CUFFT_FORWARD));
    perf.Stop();

    perf.Start("IFFT-Iteration " + std::to_string(i + 1), cols);
    CHECK_CUFFT_EXIT(cufftXtExec(plan_ifft, d_fftdata, d_odata, CUFFT_INVERSE));
    perf.Stop();
  }
  perf.PrintToCSV(output_file.c_str());

  // Copy data back to host and check accuracy
  std::vector<float> h_odata(signal_size * batch_size);
  cudaMemcpy(h_odata.data(), d_odata,
             sizeof(cufftReal) * signal_size * batch_size,
             cudaMemcpyDeviceToHost);
  checkAccuracyFloat(h_idata.data(), h_odata.data(), signal_size, batch_size,
                     1e-5);

  cufftDestroy(plan_fft);
  cufftDestroy(plan_ifft);
  cudaFree(d_idata);
  cudaFree(d_fftdata);
  cudaFree(d_odata);
}

void batchedFFT_IFFT_Float16(int signal_size, int batch_size, int iterations,
                             Perfostep &perf, const std::string &output_file) {
  __half *d_idata, *d_odata;
  __half2 *d_fftdata;
  cufftHandle plan_fft, plan_ifft;
  cudaMalloc(&d_idata, sizeof(__half) * signal_size * batch_size);
  cudaMalloc(&d_fftdata, sizeof(__half2) * (signal_size / 2 + 1) * batch_size);
  cudaMalloc(&d_odata, sizeof(__half) * signal_size * batch_size);

  // Data initialization with random data
  std::default_random_engine generator;
  std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
  std::vector<__half> h_idata(signal_size * batch_size);
  for (int i = 0; i < signal_size * batch_size; ++i) {
    h_idata[i] = __float2half(distribution(generator));
  }
  cudaMemcpy(d_idata, h_idata.data(), sizeof(__half) * signal_size * batch_size,
             cudaMemcpyHostToDevice);

  // Create FFT and IFFT plans
  CHECK_CUFFT_EXIT(cufftCreate(&plan_fft));
  CHECK_CUFFT_EXIT(cufftCreate(&plan_ifft));

  // Plan creation benchmark
  size_t workSize;
  long long int signal_size_ll = signal_size;
  ColumnNames cols = {{"Precision", "Float16"},
                      {"Signal Size", std::to_string(signal_size)},
                      {"Batch Size", std::to_string(batch_size)}};
  perf.Start("Creation_FFT", cols);
  CHECK_CUFFT_EXIT(cufftXtMakePlanMany(plan_fft, 1, &signal_size_ll, NULL, 1,
                                       signal_size_ll, CUDA_R_16F, NULL, 1,
                                       signal_size_ll / 2 + 1, CUDA_C_16F,
                                       batch_size, &workSize, CUDA_C_16F));
  perf.Stop();

  perf.Start("Creation_IFFT", cols);
  CHECK_CUFFT_EXIT(cufftXtMakePlanMany(plan_ifft, 1, &signal_size_ll, NULL, 1,
                                       signal_size_ll / 2 + 1, CUDA_C_16F, NULL,
                                       1, signal_size_ll, CUDA_R_16F,
                                       batch_size, &workSize, CUDA_C_16F));
  perf.Stop();

  // FFT and IFFT execution benchmark
  for (int i = 0; i < iterations; ++i) {
    perf.Start("FFT-Iteration " + std::to_string(i + 1), cols);
    CHECK_CUFFT_EXIT(cufftXtExec(plan_fft, d_idata, d_fftdata, CUFFT_FORWARD));
    perf.Stop();

    perf.Start("IFFT-Iteration " + std::to_string(i + 1), cols);
    CHECK_CUFFT_EXIT(cufftXtExec(plan_ifft, d_fftdata, d_odata, CUFFT_INVERSE));
    perf.Stop();
  }
  perf.PrintToCSV(output_file.c_str());

  // Copy data back to host and check accuracy
  std::vector<__half> h_odata(signal_size * batch_size);
  cudaMemcpy(h_odata.data(), d_odata, sizeof(__half) * signal_size * batch_size,
             cudaMemcpyDeviceToHost);
  checkAccuracyHalf(h_idata.data(), h_odata.data(), signal_size, batch_size,
                    1e-2);

  cufftDestroy(plan_fft);
  cufftDestroy(plan_ifft);
  cudaFree(d_idata);
  cudaFree(d_fftdata);
  cudaFree(d_odata);
}

void batchedFFT_IFFT_Float64(int signal_size, int batch_size, int iterations,
                             Perfostep &perf, const std::string &output_file) {
  cufftDoubleReal *d_idata, *d_odata;
  cufftDoubleComplex *d_fftdata;
  cufftHandle plan_fft, plan_ifft;
  cudaMalloc(&d_idata, sizeof(cufftDoubleReal) * signal_size * batch_size);
  cudaMalloc(&d_fftdata,
             sizeof(cufftDoubleComplex) * (signal_size / 2 + 1) * batch_size);
  cudaMalloc(&d_odata, sizeof(cufftDoubleReal) * signal_size * batch_size);

  // Data initialization with random data
  std::default_random_engine generator;
  std::uniform_real_distribution<double> distribution(-1.0, 1.0);
  std::vector<cufftDoubleReal> h_idata(signal_size * batch_size);
  for (int i = 0; i < signal_size * batch_size; ++i) {
    h_idata[i] = distribution(generator);
  }
  cudaMemcpy(d_idata, h_idata.data(),
             sizeof(cufftDoubleReal) * signal_size * batch_size,
             cudaMemcpyHostToDevice);

  // Create FFT and IFFT plans
  CHECK_CUFFT_EXIT(cufftCreate(&plan_fft));
  CHECK_CUFFT_EXIT(cufftCreate(&plan_ifft));

  // Plan creation benchmark
  size_t workSize;
  long long int signal_size_ll = signal_size;
  ColumnNames cols = {{"Precision", "Float64"},
                      {"Signal Size", std::to_string(signal_size)},
                      {"Batch Size", std::to_string(batch_size)}};
  perf.Start("Creation_FFT", cols);
  CHECK_CUFFT_EXIT(cufftXtMakePlanMany(plan_fft, 1, &signal_size_ll, NULL, 1,
                                       signal_size_ll, CUDA_R_64F, NULL, 1,
                                       signal_size_ll / 2 + 1, CUDA_C_64F,
                                       batch_size, &workSize, CUDA_C_64F));
  perf.Stop();

  perf.Start("Creation_IFFT", cols);
  CHECK_CUFFT_EXIT(cufftXtMakePlanMany(plan_ifft, 1, &signal_size_ll, NULL, 1,
                                       signal_size_ll / 2 + 1, CUDA_C_64F, NULL,
                                       1, signal_size_ll, CUDA_R_64F,
                                       batch_size, &workSize, CUDA_C_64F));
  perf.Stop();

  // FFT and IFFT execution benchmark
  for (int i = 0; i < iterations; ++i) {
    perf.Start("FFT-Iteration " + std::to_string(i + 1), cols);
    CHECK_CUFFT_EXIT(cufftXtExec(plan_fft, d_idata, d_fftdata, CUFFT_FORWARD));
    perf.Stop();

    perf.Start("IFFT-Iteration " + std::to_string(i + 1), cols);
    CHECK_CUFFT_EXIT(cufftXtExec(plan_ifft, d_fftdata, d_odata, CUFFT_INVERSE));
    perf.Stop();
  }

  perf.PrintToCSV(output_file.c_str());
  // Copy data back to host and check accuracy
  std::vector<cufftDoubleReal> h_odata(signal_size * batch_size);
  cudaMemcpy(h_odata.data(), d_odata,
             sizeof(cufftDoubleReal) * signal_size * batch_size,
             cudaMemcpyDeviceToHost);
  checkAccuracyDouble(h_idata.data(), h_odata.data(), signal_size, batch_size,
                      1e-10);

  cufftDestroy(plan_fft);
  cufftDestroy(plan_ifft);
  cudaFree(d_idata);
  cudaFree(d_fftdata);
  cudaFree(d_odata);
}

int main(int argc, char *argv[]) {
  argparse::ArgumentParser program("FFT Benchmark");

  program.add_argument("--size")
      .help("Size of each signal")
      .default_value(1024)
      .scan<'i', int>();

  program.add_argument("--batch")
      .help("Number of signals in the batch")
      .default_value(10)
      .scan<'i', int>();

  program.add_argument("--iterations")
      .help("Number of iterations for the benchmark")
      .default_value(1)
      .scan<'i', int>();

  program.add_argument("--output")
      .help("CSV output file")
      .default_value(std::string("fft_performance_report.csv"));

  try {
    program.parse_args(argc, argv);
  } catch (const std::exception &err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    return 1;
  }

  int signal_size = program.get<int>("--size");
  int batch_size = program.get<int>("--batch");
  int iterations = program.get<int>("--iterations");
  std::string output_file = program.get<std::string>("--output");

  Perfostep perf;

  batchedFFT_IFFT_Float32(signal_size, batch_size, iterations, perf,
                          output_file);
  batchedFFT_IFFT_Float16(signal_size, batch_size, iterations, perf,
                          output_file);
  batchedFFT_IFFT_Float64(signal_size, batch_size, iterations, perf,
                          output_file);

  return 0;
}
