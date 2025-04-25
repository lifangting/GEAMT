//
// Created by lft on 2022/11/4.
//

#ifndef DGEMT_ACO444_ANT_H
#define DGEMT_ACO444_ANT_H

#include<iostream>
#include<stdio.h>
using namespace std;
#include <cuda.h>
#include <curand_kernel.h>
#include <thrust/extrema.h>
#include "device_launch_parameters.h"
#include "cuda_runtime.h"

#define order 2

typedef struct ant
{
    int position[order];
    float fitness1;
    float fitness2;
    int isDominant;

} ant;

bool checkForError(cudaError_t error) ;

bool checkForKernelError(const char *err_msg) ;

__device__ float My_factorial(int e) ;

__global__ void g_setup_curand_states(curandState *state_d, int ant_number, unsigned long t, int THREADS);

__global__ void g_initialize_ants(ant *ants_d,  int ant_number, int THREADS) ;

__global__ void g_initialize_hormone_d(float *hormone_d, int local_snp_number, int THREADS) ;

__global__ void g_hormone_sum(int local_snp_number, float *cdf_d, float *hormone_d, float *fit_sum_d, int THREADS);

__global__ void g_cal_cdf(int local_snp_number, float *cdf_d, float *fit_sum_d, int THREADS);

__global__ void g_simulate_ants(int ant_number, int local_snp_number, ant *ants_d, curandState *state_d, float *cdf_d, int THREADS,
                                ant *resv_transfer_d, int resv_transfer_number, int i, int rank, int size, int iter) ;

__global__ void g_chiSquare_kernel(ant *ants_d, int ant_number, int **d_data, int *dc, int *d_label, int sample_number,
                                   int *select_range, int THREADS) ;

__global__ void g_K2_kernel(ant *ants_d, int ant_number, int **d_data, int *dc, int *d_label, int sample_number,
                            int *select_range, int THREADS) ;

__global__ void g_gen_select_best(ant *ants_d, int ant_number, ant *dominant_d, int *dominant_number_d, int THREADS) ;

__global__ void g_update_dominant(ant *ants_d, int ant_number, ant *dominant_d, int *dominant_number_d, int rank,int size,int iter,int gen,int THREADS);

__global__ void g_pheromone_update(ant *dominant_d, int *dominant_number_d, float *hormone_d,int THREADS) ;

bool g_calculate_fitness(ant* resv_transfer,int resv_transfer_number,int **d_data, int *dc, int *d_label, int sample_number);


bool g_execute(int max_gen, int ant_number, int local_snp_number, int sample_number, int **d_data, int *dc, int *d_label,
          int *select_range, int rank, int size, int iter, int max_iter,
          int local_coll_num, int *local_exchange_array, float *hormone_h_main,
          ant *&dominant, int &dominant_number, ant *resv_dominant, int dominant_number_all, ant *resv_transfer_mate,
          int mate_number);

#endif //DGEMT_ACO444_ANT_H
