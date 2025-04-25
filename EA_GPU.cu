//
// Created by lft on 2022/10/24.
//

#include<time.h>
#include <cstdlib>
#include <ctime>
#include<math.h>
#include <stdio.h>
#include <stdlib.h>
#include "EA_GPU.h"

#define N 99999999


bool checkForError(cudaError_t error) {
    if (error != cudaSuccess) {
        cout << cudaGetErrorString(error) << endl;
        return true;
    } else {
        return false;
    }
}

bool checkForKernelError(const char *err_msg) {
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        cout << err_msg << cudaGetErrorString(status) << endl;
        return true;
    } else {
        return false;
    }
}

__device__ float My_factorial(int e) {

    float f = 0;
    if (e > 0) {
        for (int u = 1; u <= e; u++) {
            f = f + logf(u);
        }
    }
    return f;

}

__global__ void g_setup_curand_states(curandState *state_d, int ant_number, unsigned long t, int THREADS) {

    int id = threadIdx.x + blockIdx.x * THREADS;
    if (id < ant_number) {
        curand_init(t, id, 0, &state_d[id]);
    }
}

__global__ void g_initialize_ants(ant *ants_d, int ant_number, int THREADS) {
    int id = threadIdx.x + blockIdx.x * THREADS;
    if (id < ant_number) {
        int k;
        for (k = 0; k < order; k++) {
            ants_d[id].position[k] = 0;
        }
        ants_d[id].fitness1 = 99999;
        ants_d[id].fitness2 = 99999;
        ants_d[id].isDominant = 0;
    }
}

__global__ void g_initialize_hormone_d(float *hormone_d, int local_snp_number, int THREADS) {
    int id = threadIdx.x + blockIdx.x * THREADS;
    if (id < local_snp_number) {
        hormone_d[id] = 0.5;
    }
}

__global__ void g_hormone_sum(int local_snp_number, float *cdf_d, float *hormone_d, float *fit_sum_d, int THREADS) {
    int id = threadIdx.x + blockIdx.x * THREADS;
    if (id < local_snp_number) {
        float sum = 0;
        for (int i = 0; i <= id; i++) {
            sum = sum + hormone_d[i];
        }
        cdf_d[id] = sum;
        if (id == (local_snp_number - 1)) {
            *fit_sum_d = sum;
        }
    }
}

__global__ void g_cal_cdf(int local_snp_number, float *cdf_d, float *fit_sum_d, int THREADS) {
    int id = threadIdx.x + blockIdx.x * THREADS;
    if (id < local_snp_number) {
        cdf_d[id] /= *fit_sum_d;
    }
}


__global__ void g_simulate_ants(int ant_number, int local_snp_number, ant *ants_d, curandState *state_d, float *cdf_d, int THREADS,
                int gen, int rank, int size, int iter,ant* mate_d, int mate_number) {
    int id = threadIdx.x + blockIdx.x * THREADS;

    if (id < ant_number) {
        if(rank != size - 1 )
            mate_number=0;
        if (rank == size - 1 && iter != 0 && gen == 0 && id < mate_number) {
            for (int i = 0; i < order; i++) {
                ants_d[id].position[i] = mate_d[id].position[i];
            }
            ants_d[id].isDominant = 0;

        } else {
        for (int i = 0; i < order; i++) {
            float x = (double) (curand(&state_d[id]) % 100000) / 100000.0;
            for (int j = 0; j < local_snp_number; j++) {
                if (x < cdf_d[j]) {
                    ants_d[id].position[i] = j;
                    //de-duplicate
                    for (int h = 0; h < i; h++) {
                        while (ants_d[id].position[h] == ants_d[id].position[i]) {
                            ants_d[id].position[i] = curand(&state_d[id]) % local_snp_number;
                        }
                    }
                    break;
                }
            }
        }
        ants_d[id].isDominant = 0;
    }
}

}

__global__ void g_chiSquare_kernel(ant *ants_d, int ant_number, int **d_data, int *dc, int *d_label, int sample_number,
                                   int *select_range, int THREADS) {
    int tid = threadIdx.x + blockIdx.x * THREADS;

    if (tid < ant_number) {

        int m;
        int tmp = powf(3, order);
        int *case_observed = new int[tmp];
        int *control_observed = new int[tmp];
        float *case_expected = new float[tmp];
        float *control_expected = new float[tmp];
        for (int i = 0; i < tmp; i++) {
            case_observed[i] = 0;
            control_observed[i] = 0;
            case_expected[i] = 0;
            control_expected[i] = 0;
        }


        int casecount = 0;
        int controlcount = 0;

        for (int i = 0; i < sample_number; i++) {
            m = 0;
            for (int j = 0; j < order; j++) {
                m = m * 3 + d_data[i][select_range[ants_d[tid].position[j]]];

            }
            if (d_label[i] == 1) {
                case_observed[m] = case_observed[m] + 1;
                casecount++;
                //cout<<case_observed[0]<<endl;
            } else {
                control_observed[m] = control_observed[m] + 1;
                controlcount++;
                //cout<<case_observed[1]<<endl;
            }
        }

        for (int i = 0; i < tmp; i++) {
            case_expected[i] = (case_observed[i] + control_observed[i]) * casecount / (float) sample_number;
            control_expected[i] = (case_observed[i] + control_observed[i]) * controlcount / (float) sample_number;
        }

        float X2 = 0;
        for (int m = 0; m < tmp; m++) {
            if (case_expected[m] != 0)
                X2 = X2 + (case_expected[m] - case_observed[m]) * (case_expected[m] - case_observed[m]) /
                          (float) case_expected[m];
        }

        for (int n = 0; n < tmp; n++) {
            if (control_expected[n] != 0)
                X2 = X2 + (control_expected[n] - control_observed[n]) * (control_expected[n] - control_observed[n]) /
                          (float) control_expected[n];
        }
        //return X2;
        ants_d[tid].fitness2 = -X2;

//        printf("%d\t%d\tK2=%f\tX2=%f\n", select_range[ants_d[tid].position[0]], select_range[ants_d[tid].position[1]], ants_d[tid].fitness1,ants_d[tid].fitness2);


        delete[] case_observed;
        delete[] case_expected;
        delete[] control_observed;
        delete[] control_expected;
    }
}


__global__ void g_MI_kernel(ant *ants_d, int ant_number, int **d_data, int *dc, int *d_label, int sample_number,
                                   int *select_range, int THREADS) {
    int tid = threadIdx.x + blockIdx.x * THREADS;

    if (tid < ant_number) {
        int m;
        int tmp = powf(3, order);
        int *case_observed = new int[tmp];
        int *control_observed = new int[tmp];
        int casecount = 0;
        int controlcount = 0;
        float Entroy_label = 0.0;

        float *p_case=new float[tmp];
        float *p_control=new float[tmp];
        float tmp2 = 0.0;
        float MI = 0.0;

        for (int i = 0; i < tmp; i++) {
            case_observed[i] = 0;
            control_observed[i] = 0;
        }


        for (int i = 0; i < sample_number; i++) {
            m = 0;
            for (int j = 0; j < order; j++) {
                m = m * 3 + d_data[i][select_range[ants_d[tid].position[j]]];
            }
            if (d_label[i] == 1) {
                case_observed[m] = case_observed[m] + 1;
                casecount++;
            } else {
                control_observed[m] = control_observed[m] + 1;
                controlcount++;
            }
        }

        float p_case_label = (float)casecount / sample_number;
        float p_control_label = (float)controlcount / sample_number;



//3.Compute Mutual Information  I(X;Y)=H(Y)-H(Y|X)     H(Y|X)= sum_x sum_y  -p(x)p(y|x)logp(y|x)

        Entroy_label = -(p_case_label * logf(p_case_label) + p_control_label * logf(p_control_label));

        for (int l =0;l< tmp;l++) {
            if ((case_observed[l] + control_observed[l]) != 0) {
                p_case[l] = (float)case_observed[l]/ (case_observed[l] + control_observed[l]);
                p_control[l] = (float)control_observed[l] / (case_observed[l] + control_observed[l]);

            } else {
                p_case[l] = 0;
                p_control[l] = 0;
            }
            if (p_case[l] != 0)
                tmp2 = tmp2 + p_case[l] * logf(p_case[l]) * case_observed[l] / sample_number;
            if (p_control[l] != 0)
                tmp2 = tmp2 + p_control[l] * logf(p_control[l]) * control_observed[l]/ sample_number;

        }

        tmp2 = -tmp2;
        MI = Entroy_label - tmp2;
        ants_d[tid].fitness2 = -MI;

//        printf("%f-- ",ants_d[tid].fitness2);



        delete[] case_observed;
        delete[] control_observed;
        delete[] p_case;
        delete[] p_control;

    }
}

__global__ void g_K2_kernel(ant *ants_d, int ant_number, int **d_data, int *dc, int *d_label, int sample_number,
                            int *select_range, int THREADS) {
    int tid = threadIdx.x + blockIdx.x * THREADS;

    if (tid < ant_number) {

        int m;
        int tmp = powf(3, order);
        int *case_observed = new int[tmp];
        int *control_observed = new int[tmp];
        for (int i = 0; i < tmp; i++) {
            case_observed[i] = 0;
            control_observed[i] = 0;
        }


        int casecount = 0;
        int controlcount = 0;

        for (int i = 0; i < sample_number; i++) {
            m = 0;
            for (int j = 0; j < order; j++) {
                m = m * 3 + d_data[i][select_range[ants_d[tid].position[j]]];

            }
            if (d_label[i] == 1) {
                case_observed[m] = case_observed[m] + 1;
                casecount++;
                //cout<<case_observed[0]<<endl;
            } else {
                control_observed[m] = control_observed[m] + 1;
                controlcount++;
                //cout<<case_observed[1]<<endl;
            }
        }

        float y = 0;
        float z = 0;
        float r = 0;
        for (int i = 0; i < tmp; i++) {
            y = My_factorial(case_observed[i] + control_observed[i] + 1);
            r = My_factorial(case_observed[i]) + My_factorial(control_observed[i]);
            z = z + (r - y);
        }

        if (z < 0) {
            z = -z;
        }


        ants_d[tid].fitness1 = z;

//        printf("%d\t%d\tK2=%f\n", select_range[ants_d[tid].position[0]], select_range[ants_d[tid].position[1]], ants_d[tid].fitness);

        delete[] case_observed;
        delete[] control_observed;
    }


}

__global__ void
g_gen_select_best(ant *ants_d, int ant_number, ant *dominant_d, int *dominant_number_d, int gen, int iter, int rank,
                  int size, int THREADS) {
    int id = threadIdx.x + blockIdx.x * THREADS;


    if (id < ant_number) {
        if ((rank != size - 1) && (gen == 0) || (rank == size - 1) && (iter == 0) && (gen == 0))
            *dominant_number_d = 0;


        ants_d[id].isDominant = 1;

        for (int i = 0; i < ant_number; i++) {
            if (ants_d[i].fitness1 <= ants_d[id].fitness1 && ants_d[i].fitness2 <= ants_d[id].fitness2
                && (ants_d[i].fitness1 < ants_d[id].fitness1 || ants_d[i].fitness2 < ants_d[id].fitness2)) {
                ants_d[id].isDominant = 0;
                break;
            }
        }


        if (ants_d[id].isDominant == 1) {
            for (int i = 0; i < (*dominant_number_d); i++) {
                int count = 0;
                if (dominant_d[i].fitness1 <= ants_d[id].fitness1 && dominant_d[i].fitness2 <= ants_d[id].fitness2
                    && (dominant_d[i].fitness1 < ants_d[id].fitness1 || dominant_d[i].fitness2 < ants_d[id].fitness2)) {
                    ants_d[id].isDominant = 0;
                    break;
                }
                for (int j = 0; j < order; j++) {
                    if (ants_d[id].position[j] == dominant_d[i].position[j]) {
                        count++;
                    }
                }
                if (count == order)
                    ants_d[id].isDominant = 0;
            }
            for (int i = 0; i < (*dominant_number_d); i++) {
                if (ants_d[id].fitness1 <= dominant_d[i].fitness1 && ants_d[id].fitness2 <= dominant_d[i].fitness2
                    && (ants_d[id].fitness1 < dominant_d[i].fitness1 || ants_d[id].fitness2 < dominant_d[i].fitness2)) {
                    dominant_d[i].isDominant = 0; 
                }

            }
        }

//        printf("rank=%d,iter=%d,gen=%d,%d\t%d\tK2=%f\tX2=%f\tisDominate=%d\t上一代的Dominate_number=%d\n", rank,iter,gen,ants_d[id].position[0], ants_d[id].position[1],
//               ants_d[id].fitness1, ants_d[id].fitness2, ants_d[id].isDominant,(*dominant_number_d));


    }
}


__global__ void
g_update_dominant(ant *ants_d, int ant_number, ant *dominant_d, int *dominant_number_d, int rank, int size, int iter,
                  int gen, int THREADS) {
    int id = threadIdx.x + blockIdx.x * THREADS;
    if (id == 0) {
        //printf("rank=%d\t上一代的dominant_number_d=%d\n", rank,(*dominant_number_d));


        //1.更新dominant_number_d
        int dominant_number_old = (*dominant_number_d);
        int count = 0;
        for (int i = 0; i < ant_number; i++) {
            if (ants_d[i].isDominant == 1) {
                for (int m = 0; m < order; m++)
                    dominant_d[(*dominant_number_d)].position[m] = ants_d[i].position[m];
                dominant_d[(*dominant_number_d)].fitness1 = ants_d[i].fitness1;
                dominant_d[(*dominant_number_d)].fitness2 = ants_d[i].fitness2;
                dominant_d[(*dominant_number_d)].isDominant = 1;
                (*dominant_number_d)++;
                count++;
            }
        }
        for (int i = 0; i < dominant_number_old; i++)
            if (dominant_d[i].isDominant == 0)
                (*dominant_number_d)--;


        //2.Update dominant_d
        if (dominant_number_old != 0) {
            for (int i = 0; i < dominant_number_old + count - 1; i++) {
                for (int j = i + 1; j < dominant_number_old + count; j++) {
                    if (dominant_d[i].isDominant < dominant_d[j].isDominant) {
                        for (int m = 0; m < order; m++)
                            dominant_d[i].position[m] = dominant_d[j].position[m];
                        dominant_d[i].fitness1 = dominant_d[j].fitness1;
                        dominant_d[i].fitness2 = dominant_d[j].fitness2;
                        dominant_d[i].isDominant = 1;
                        dominant_d[j].isDominant = 0;
//                        break;

                    }

                }
            }
        }
    }

    //region 
//             for (int i = 0; i < dominant_number_old; i++) {
//                    for(int j=0;j<ant_number;j++){
//                        if (ants_d[j].isDominant == 1&&ants_d[j].fitness1 <= dominant_d[i].fitness1 && ants_d[j].fitness2 <= dominant_d[i].fitness2 &&
//                            (ants_d[j].fitness1 < dominant_d[i].fitness1 || ants_d[j].fitness2 < dominant_d[i].fitness2))
//                        {
//                            dominant_d[i].isDominant=0;
//                            (*dominant_number_d) = (*dominant_number_d) - 1;
//                            break;
//                        }
//
//                    }
//                }
    //endregion

    //printf("rank=%d\t这一代的dominant_number_d=%d\n", rank,(*dominant_number_d));
}


__global__ void g_pheromone_update(ant *ants_d, int ant_number,int* count_tmp,int snp_number,ant *dominant_d, int *dominant_number_d, float *hormone_d, int THREADS) {

    int id = threadIdx.x + blockIdx.x * THREADS;

    if (id == 0) {
        for (int u = 0; u < snp_number; u++) {
            count_tmp[u] = 0;
        }
//        for(int c=0;c<ant_number;c++){
//            for (int j = 0; j < order; j++) {
//                if(count_tmp[ants_d[c].position[j]]==0) {
//                    hormone_d[ants_d[c].position[j]]=hormone_d[ants_d[c].position[j]]*0.98;
//                    if(hormone_d[ants_d[c].position[j]]<5)
//                        hormone_d[ants_d[c].position[j]]=5;
//                    count_tmp[ants_d[c].position[j]] = 1;
//                }
//            }
//        }

        for (int i = 0; i < (*dominant_number_d); i++) {
            for (int j = 0; j < order; j++) {
                hormone_d[dominant_d[i].position[j]] = hormone_d[dominant_d[i].position[j]] + 0.05;
                if (hormone_d[dominant_d[i].position[j]] > 2)
                    hormone_d[dominant_d[i].position[j]] = 2;
            }
        }
    }
}

__global__ void g_pheromone_update_main(int *count_tmp, int snp_number, ant *dominant_d, int *dominant_number_d, ant *mate_d,
                        int mate_number,
                        float *hormone_d, int THREADS) {

    int id = threadIdx.x + blockIdx.x * THREADS;

    if (id == 0) {
        for (int u = 0; u < snp_number; u++) {
            count_tmp[u] = 0;
        }

        for (int i = 0; i < (*dominant_number_d); i++) {
            for (int j = 0; j < order; j++)
                count_tmp[dominant_d[i].position[j]]++;
        }
        for (int i = 0; i < mate_number; i++) {
            for (int j = 0; j < order; j++)
                count_tmp[dominant_d[i].position[j]]++;
        }

        for (int m = 0; m < snp_number; m++) {
            hormone_d[m] = (1 + 0.05*  count_tmp[m]) * hormone_d[m];
            if (hormone_d[m] > 10)
                hormone_d[m] = 10;
        }

    }
}

__global__ void g_show_pheromone(int local_snp_number, float *hormone_d, int THREADS) {

    int id = threadIdx.x + blockIdx.x * THREADS;
    if (id == 0) {
        for (int u = 0; u < local_snp_number; u++)
            printf("%.1lf ", hormone_d[u]);
        printf("\n--------------------------------------------------------------\n");
    }
}

//__global__ void g_show_solution(int ant_number, int *dominant_number_d, ant *dominant_d, int THREADS) {
//
//    int id = threadIdx.x + blockIdx.x * THREADS;
//    if (id < (*dominant_number_d)) {
//        printf("dominant_number_d=%d,position=%d %d,f1=%f,f2=%f,isD=%d\n", (*dominant_number_d),
//               dominant_d[id].position[0], dominant_d[id].position[1],
//               dominant_d[id].fitness1, dominant_d[id].fitness2, dominant_d[id].isDominant);
//    }
//}


__global__ void g_show_solution2(int ant_number, int *dominant_number_d, ant *dominant_d, int THREADS) {

    int id = threadIdx.x + blockIdx.x * THREADS;
    if (id < (*dominant_number_d)) {
        printf("dominant_number_d=%d,position=%d %d,f1=%f,f2=%f,isD=%d##################\n", (*dominant_number_d),
               dominant_d[id].position[0], dominant_d[id].position[1],
               dominant_d[id].fitness1, dominant_d[id].fitness2, dominant_d[id].isDominant);
    }
}

__global__ void g_show_solution3(int ant_number, int *dominant_number_d, ant *dominant_d, int THREADS) {

    int id = threadIdx.x + blockIdx.x * THREADS;
    if (id < (*dominant_number_d)) {
        printf("dominant_number_d=%d,position=%d %d,f1=%f,f2=%f,isD=%d^^^^^^^^^^^^^^^^\n", (*dominant_number_d),
               dominant_d[id].position[0], dominant_d[id].position[1],
               dominant_d[id].fitness1, dominant_d[id].fitness2, dominant_d[id].isDominant);
    }
}


bool g_execute(int max_gen, int ant_number, int local_snp_number, int sample_number, int **d_data, int *dc, int *d_label,
          int *select_range, int rank, int size, int iter, int max_iter,
          int local_coll_num, int *local_exchange_array, float *hormone_h_main,
          ant *&dominant, int &dominant_number, ant *resv_dominant, int dominant_number_all, ant *resv_transfer_mate,
          int mate_number) {


    // region 1.Setting the block and thread
    int BLOCKS, THREADS, BLOCKS2, THREADS2;
    if (ant_number <= 512) {
        BLOCKS = 1;
        THREADS = ant_number;
    } else {
        THREADS = 512;
        BLOCKS = ceil(ant_number / (float) THREADS);
    }

    if (local_snp_number <= 512) {
        BLOCKS2 = 1;
        THREADS2 = local_snp_number;
    } else {
        THREADS2 = 512;
        BLOCKS2 = ceil(local_snp_number / (float) THREADS);
    }
    // endregion


    // region 2.Allocating memory and copying
    bool error;
    curandState *state_d;
    ant *ants_d;
    float *hormone_d;
    float *cdf_d;
    float *fit_sum_d;
    int *select_range_d;
    int *count_tmp;
    ant *dominant_d;
    int *dominant_number_d;
    ant *mate_d;

    //Allocating memory
    error = checkForError(cudaMalloc((void **) &state_d, ant_number * sizeof(curandState)));
    if (error) { return true; }
    error = checkForError(cudaMalloc((void **) &ants_d, ant_number * sizeof(ant)));
    if (error) { return true; }

    error = checkForError(cudaMalloc((void **) &hormone_d, local_snp_number * sizeof(float)));
    if (error) { return true; }

    error = checkForError(cudaMalloc((void **) &cdf_d, local_snp_number * sizeof(float)));
    if (error) { return true; }
    error = checkForError(cudaMalloc((void **) &fit_sum_d, sizeof(float)));
    if (error) { return true; }
    error = checkForError(cudaMalloc((void **) &select_range_d, sizeof(int) * local_snp_number));
    if (error) { return true; }

    error = checkForError(cudaMalloc((void **) &dominant_d, ant_number * sizeof(ant)));
    if (error) { return true; }
    error = checkForError(cudaMalloc((void **) &dominant_number_d, sizeof(int)));
    if (error) { return true; }

    error = checkForError(cudaMalloc((void **) &count_tmp, local_snp_number * sizeof(int)));
    if (error) { return true; }

    if ((rank == size - 1) && iter != 0) {
        error = checkForError(cudaMalloc((void **) &mate_d, mate_number * sizeof(ant)));
        if (error) { return true; }

    }


    //Copying
    error = checkForError(
            cudaMemcpy(select_range_d, select_range, local_snp_number * sizeof(int), cudaMemcpyHostToDevice));
    if (error) { return true; }

    if (rank == size - 1) {
        error = checkForError(cudaMemcpy(dominant_number_d, &dominant_number_all, sizeof(int), cudaMemcpyHostToDevice));
        if (error) { return true; }
    }

    if ((rank == size - 1) && iter != 0) {
        //hormone_d=hormone_h_main;
        error = checkForError(
                cudaMemcpy(hormone_d, hormone_h_main, local_snp_number * sizeof(float), cudaMemcpyHostToDevice));
        if (error) { return true; }

        error = checkForError(
                cudaMemcpy(dominant_d, resv_dominant, dominant_number_all * sizeof(ant), cudaMemcpyHostToDevice));
        if (error) { return true; }


        error = checkForError(
                cudaMemcpy(mate_d, resv_transfer_mate, mate_number * sizeof(ant), cudaMemcpyHostToDevice));
        if (error) { return true; }


    }
    // endregion


    // region 3.Population Initialization
    time_t t;
    time(&t);
    g_setup_curand_states <<< BLOCKS, THREADS >>>(state_d, ant_number, (unsigned long) t, THREADS);
    cudaDeviceSynchronize();
    if (checkForKernelError("g_setup_curand_states is failing ")) return true;

    g_initialize_ants <<< BLOCKS, THREADS >>>(ants_d, ant_number, THREADS);
    cudaDeviceSynchronize();
    if (checkForKernelError("g_initialize_ants is failing ")) return true;


    if ((rank != size - 1) || ((rank == size - 1) && (iter == 0))) {

        g_initialize_hormone_d<<< BLOCKS2, THREADS2 >>>(hormone_d, local_snp_number, THREADS);
        cudaDeviceSynchronize();
        if (checkForKernelError("g_initialize_hormone_d is failing ")) return true;

        g_initialize_ants <<< BLOCKS, THREADS >>>(dominant_d, ant_number, THREADS);
        cudaDeviceSynchronize();
        if (checkForKernelError("g_initialize_genbest is failing ")) return true;
    }



    if ((rank == size - 1) && (iter != 0)) {
        g_pheromone_update_main<<< BLOCKS2, THREADS2 >>>(count_tmp, local_snp_number, dominant_d, dominant_number_d,
                                                         mate_d,
                                                         mate_number, hormone_d, THREADS);
        cudaDeviceSynchronize();
        if (checkForKernelError("g_pheromone_update is failing ")) return true;
    }


    // endregion


    // region 4.Loop
    for (int i = 0; i < max_gen; i++) {

        g_hormone_sum<<< BLOCKS2, THREADS2 >>>(local_snp_number, cdf_d, hormone_d, fit_sum_d, THREADS);
        cudaDeviceSynchronize();
        if (checkForKernelError("g_hormone_sum is failing ")) return true;

        g_cal_cdf<<< BLOCKS2, THREADS2 >>>(local_snp_number, cdf_d, fit_sum_d, THREADS);
        cudaDeviceSynchronize();
        if (checkForKernelError("g_cal_cdf is failing ")) return true;

        g_simulate_ants<<< BLOCKS, THREADS >>>(ant_number, local_snp_number, ants_d, state_d, cdf_d, THREADS,
                                               i, rank, size, iter,mate_d, mate_number);
        cudaDeviceSynchronize();
        if (checkForKernelError("g_simulate_ants is failing ")) return true;

        g_K2_kernel<<< BLOCKS, THREADS >>>(ants_d, ant_number, d_data, dc, d_label, sample_number,
                                           select_range_d, THREADS);
        cudaDeviceSynchronize();
        if (checkForKernelError("g_K2_kernel is failing ")) return true;

        g_MI_kernel<<< BLOCKS, THREADS >>>(ants_d, ant_number, d_data, dc, d_label, sample_number,
                                                  select_range_d, THREADS);
        cudaDeviceSynchronize();
        if (checkForKernelError("g_MI_kernel is failing ")) return true;

        g_gen_select_best<<< BLOCKS, THREADS >>>(ants_d, ant_number, dominant_d, dominant_number_d, i, iter, rank, size,
                                                 THREADS);
        cudaDeviceSynchronize();
        if (checkForKernelError(" g_gen_select_best is failing ")) return true;


//        if (rank == size - 1&& i==0) {
//            g_show_solution<<< BLOCKS, THREADS >>>(ant_number, dominant_number_d, dominant_d, THREADS);
//        }

        g_update_dominant<<<1, 1>>>(ants_d, ant_number, dominant_d, dominant_number_d, rank, size, iter, i, THREADS);
        cudaDeviceSynchronize();
        if (checkForKernelError("g_update_dominant is failing ")) return true;


//        if (rank == size - 1&& i==0) {
//            g_show_solution2<<< BLOCKS, THREADS >>>(ant_number, dominant_number_d, dominant_d, THREADS);
//        }


        g_pheromone_update<<< BLOCKS2, THREADS2 >>>(ants_d, ant_number, count_tmp,local_snp_number,dominant_d, dominant_number_d, hormone_d, THREADS);
        cudaDeviceSynchronize();
        if (checkForKernelError("g_pheromone_update is failing ")) return true;

//        if (i >= 990) {
//            g_show_solution<<< BLOCKS, THREADS >>>(dominant_number_d, dominant_d,select_range_d, rank,THREADS);
//            cudaDeviceSynchronize();
//        }

    }
    // endregion



    //region 5.Copy Back
    error = checkForError(cudaMemcpy(&dominant_number, dominant_number_d, sizeof(int), cudaMemcpyDeviceToHost));
    if (error) { return true; }
//    dominant = (ant *) malloc(dominant_number * sizeof(ant));
    error = checkForError(cudaMemcpy(dominant, dominant_d, dominant_number * sizeof(ant), cudaMemcpyDeviceToHost));
    if (error) { return true; }

    for (int l = 0; l < dominant_number; l++)
        for (int m = 0; m < order; m++)
            dominant[l].position[m] = select_range[dominant[l].position[m]];

    if (rank == size - 1) {
        // hormone_h_main = hormone_d;
        if (checkForError(
                cudaMemcpy(hormone_h_main, hormone_d, local_snp_number * sizeof(float), cudaMemcpyDeviceToHost)));
        if (error) { return true; }
    }
    //endregion

    //region 6.Sort SNPs by pheromone, fill in local_array
    if (rank != size - 1) {
        float hormone_h[local_snp_number];
        double prob[local_snp_number];
        double wheel[local_snp_number];

        cudaMemcpy(hormone_h, hormone_d, local_snp_number * sizeof(float), cudaMemcpyDeviceToHost);

        //抽样方法
        double max = 1;
        double min = 999;
        double sum = 0;



        for (int i = 0; i < local_snp_number; i++) {
            if (hormone_h[i] > max) {
                max = hormone_h[i];
            }
            if (hormone_h[i] < min) {
                min = hormone_h[i];
            }
        }

        for (int i = 0; i < local_snp_number; i++) {
           prob[i] = exp(min+(max-hormone_h[i]));

//            prob[i] = 1/(1+exp(-hormone_h[i]));
            // prob[i] = hormone_h[i];

            sum = sum + prob[i];
        }
        for (int i = 0; i < local_snp_number; i++) {
            prob[i] = prob[i] / sum;
        }


        wheel[0] = prob[0];
        for (int i = 1; i < local_snp_number - 1; i++) {
            wheel[i] = prob[i] + wheel[i - 1];
        }
        wheel[local_snp_number - 1] = 1.000000;

        int flag[local_snp_number];
        for (int o = 0; o < local_snp_number; o++) {
            flag[o] = 0;
        }

        srand(time(NULL));
        int u = 0;
        double p;
        while (u < local_coll_num) {
            p = (double) (rand() % (N + 1)) / (double) (N + 1);
            for (int i = 0; i < local_snp_number; i++) {
                if (p < wheel[i] && flag[i] == 0) {
                    local_exchange_array[u] = select_range[i];
                    flag[i] = 1;
                    u = u + 1;
                    break;
                } else if (p < wheel[i] && flag[i] == 1) {
                    break;
                }
            }
        }
        int tmp[local_snp_number];
        for (int d = 0; d < local_snp_number; d++) {
            tmp[d] = select_range[d];
        }
        int head = 0;
        int tail = local_snp_number - 1;
        for (int d = 0; d < local_snp_number; d++) {
            if (flag[d] == 1) {
                select_range[head] = tmp[d];
                head++;
            } else {
                select_range[tail] = tmp[d];
                tail--;
            }
        }


        for (int i = 0; i < local_coll_num; i++) {
            local_exchange_array[i] = select_range[i];
        }

    }
//    endregion




    //region 7.Free Memory
    cudaFree(ants_d);
    cudaFree(state_d);
    cudaFree(hormone_d);
    cudaFree(cdf_d);
    cudaFree(fit_sum_d);
    cudaFree(select_range_d);
    cudaFree(dominant_d);
    cudaFree(dominant_number_d);
    if ((rank == size - 1) && iter != 0)
        cudaFree(mate_d);




    //endregion

    return false;

}

