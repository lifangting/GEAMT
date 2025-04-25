#include <mpi.h>
#include <iostream>
#include <algorithm>
#include <string>
#include <stdio.h>
#include<time.h>
#include <math.h>
#include <vector>
#include "utils.h"
#include "EA_GPU.h"
#include<bits/stdc++.h>


using namespace std;

void Rangecal(int snp_number, int rank, int size, int &start, int &end) {
    int tmp = snp_number - (snp_number / size) * size;
    if (rank < tmp) {
        start = (snp_number / size + 1) * rank;
        end = start + snp_number / size;
    } else {
        start = (snp_number / size + 1) * tmp + snp_number / size * (rank - tmp);
        end = start + snp_number / size - 1;
    }
}

void Regroup(int a[], int b[], int part_number, int coll_number) {
    //MakeRand(a,snp_number);
    int index = 0;
    int p_index;
    int count[10];
    for (int i = 0; i < part_number; i++)
        count[i] = 0;
    for (int i = 0; i < part_number; i++) {
        p_index = i;
        for (int j = 0; j < coll_number / part_number; j++) {
            index = i * coll_number / part_number + j;
            p_index = (p_index + 1) % part_number;
            if (p_index == i) {
                p_index = (p_index + 1) % part_number;
            }
            int m = p_index * coll_number / part_number + count[p_index];
            if (index < coll_number)
                b[index] = a[m];
            count[p_index]++;
        }
    }
}


int main(int argc, char *argv[]) {

    //region 1.Enable MPI distribution, setup GPUs
    int rank, size, GPU_num;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    cudaGetDeviceCount(&GPU_num);
    cudaSetDevice(rank % GPU_num);
    cudaDeviceReset();
    //endregion


    //region 2.Defining new MPI data types
    ant *example;
    example = (ant *) malloc(sizeof(ant));
    MPI_Datatype MPI_ant;
    MPI_Datatype T[] = {MPI_INT, MPI_FLOAT, MPI_FLOAT, MPI_INT};
    int count = 4;
    int B[] = {order, 1, 1, 1};
    MPI_Aint disp[5];
    MPI_Aint offsets[4];
    MPI_Get_address(example, &disp[0]);
    MPI_Get_address(&example->position, &disp[1]);
    MPI_Get_address(&example->fitness1, &disp[2]);
    MPI_Get_address(&example->fitness2, &disp[3]);
    MPI_Get_address(&example->isDominant, &disp[4]);
    offsets[0] = disp[1] - disp[0];
    offsets[1] = disp[2] - disp[0];
    offsets[2] = disp[3] - disp[0];
    offsets[3] = disp[4] - disp[0];
    MPI_Type_create_struct(count, B, offsets, T, &MPI_ant);
    MPI_Type_commit(&MPI_ant);
    //endregion


    //region 3.Construct two process groups from the world_group process group
    MPI_Group world_group;
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);

    //Please note: Replace the number `4` in the code `const int ranks[1] = {4};` with `size - 1`.
    const int ranks[1] = {4}; //Replace with size-1 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    int n = 1;
    MPI_Group group1, group2;
    MPI_Group_excl(world_group, n, ranks, &group1);
    MPI_Group_incl(world_group, n, ranks, &group2);
    MPI_Comm comm1, comm2;  // 根据group1 group2分别构造两个通信域
    MPI_Comm_create(MPI_COMM_WORLD, group1, &comm1);
    MPI_Comm_create(MPI_COMM_WORLD, group2, &comm2);
    //endregion


    //region 4 ACO, model, dataset setup
    int max_iter = 6;
    int max_gen = 50;
    int ant_number = 1024;
    int dataset_Number = 50;
    string model_Index = "5";
    int sample_number;
    int snp_number;
    int **d_data = NULL;
    int *d_label = NULL;
    int *dc = NULL;
    int Right_Dataset_Num = 0;
    double start1, start2, end1, end2, cost;
    double time_sum;
    double timestore[dataset_Number];

    //endregion

    //region 5 Define variable
    //Information Transfer Variables
    int *displs, *recvCount;
    displs = (int *) malloc(sizeof(int) * size);
    recvCount = (int *) malloc(sizeof(int) * size);
    ant *dominant = NULL;
    int dominant_number = 0;
    ant *resv_dominant_old = NULL;
    ant *resv_dominant = NULL;

    ant *resv_transfer_mate = NULL;
    int mate_number = 0;
    //Task update variables
    int local_snp_number;
    int local_coll_num;
    int *recv_snp = NULL;
    float *hormone_h_main = NULL;
    int obj1, obj2;

    //endregion


    //region 4.For each dataset, run the algorithm
    for (int dataset_index = 1; dataset_index <= dataset_Number; dataset_index++) {
        start2 = clock();

        //region 4.1 Reading a dataset
        char str[10];
        vector<string> snp_names;
        int dominant_number_all = 0;
        sprintf(str, "%d", dataset_index);
        string dataset_name = "";
        if (dataset_index < 10) {
            dataset_name = "model" + model_Index + "_EDM-1_0" + str + ".txt";
        } else if (dataset_index < 100) {
            dataset_name = "model" + model_Index + "_EDM-1_" + str + ".txt";
        } else {
            dataset_name = "model" + model_Index + "_EDM-1_" + str + ".txt";
        }
        string filename = "/code/NMF1000SNP4000SAM/model" + model_Index + "_EDM-1/" + dataset_name;
//        string filename = "/code/Data/AMD_Right.txt";

        if (rank == size - 1) {
            cout << "Model" + model_Index + "----Dataset" + str + "------------------------------------------------" << endl;
        }
        g_read_data(filename, d_data, d_label, dc, sample_number, snp_number, snp_names, obj1, obj2);
        //endregion

        //region 4.2 Initial grouping of main and auxiliary tasks
        int start, end;
        if (rank != size - 1) {
            Rangecal(snp_number, rank, size - 1, start, end);
        } else {
            start = 0;
            end = snp_number - 1;
        }
        local_snp_number = end - start + 1;
        int select_range[local_snp_number];
        for (int i = 0; i < end - start + 1; i++)
            select_range[i] = start + i;
        //endregion


        //region 4.3 Preparation for task update
        local_coll_num = snp_number / (size - 1)*0.9 ;
        int local_exchange_array[local_coll_num];
        if (rank == 0)
            recv_snp = (int *) calloc((size - 1) * local_coll_num, sizeof(int));
        //endregion
        if (rank == size - 1) {
            resv_dominant_old = (ant *) calloc(ant_number, sizeof(ant));
            resv_dominant = (ant *) calloc(ant_number, sizeof(ant));
            resv_transfer_mate = (ant *) calloc(100, sizeof(ant));
            hormone_h_main = (float *) calloc(local_snp_number, sizeof(float));
        }
        dominant = (ant *) calloc(ant_number, sizeof(ant));


        //region 4.6 Start the outer loop
        for (int iter = 0; iter < max_iter; iter++) {
            //region Start the inner loop
            if (g_execute(max_gen, ant_number, local_snp_number, sample_number, d_data, dc, d_label, select_range,
                          rank, size, iter, max_iter,
                          local_coll_num, local_exchange_array, hormone_h_main, dominant, dominant_number,
                          resv_dominant, dominant_number_all, resv_transfer_mate,
                          mate_number)) {
                cout << "rank="<<rank<<" GPU Related error" << endl;
            }
            //endregion

            //region 4.6.1 Information transfer module
            //region Collect the optimal solutions for each task
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Gather(&dominant_number, 1, MPI_INT, recvCount, 1, MPI_INT, size - 1, MPI_COMM_WORLD);

            if (rank == size - 1) {
                displs[0] = 0;
                for (int b = 1; b < size; b++) {
                    displs[b] = displs[b - 1] + recvCount[b - 1];
                }
                dominant_number_all = displs[size - 1] + recvCount[size - 1];
//                resv_dominant_old = (ant *) malloc((dominant_number_all) * sizeof(ant));
//                if(dominant_number_all>=ant_number)
            }
            MPI_Gatherv(dominant, dominant_number, MPI_ant, resv_dominant_old, recvCount, displs, MPI_ant, size - 1,
                        MPI_COMM_WORLD);
            //endregion

            //region Perform global non-dominated sorting
            if (rank == size - 1) {
                int count;
                count = dominant_number_all;
                int flag[count];
                for (int i = 0; i < count; i++) {
                    flag[i] = 1;
                }


                for (int b = 0; b < count - 1; b++) {
                    for (int d = b + 1; d < count; d++) {
                        if (flag[b] == 1 && flag[d] == 1 &&
                            resv_dominant_old[b].fitness1 <= resv_dominant_old[d].fitness1 &&
                            resv_dominant_old[b].fitness2 <= resv_dominant_old[d].fitness2 &&
                            (resv_dominant_old[b].fitness1 < resv_dominant_old[d].fitness1 ||
                             resv_dominant_old[b].fitness2 < resv_dominant_old[d].fitness2)) {
                            flag[d] = 0;
                            dominant_number_all--;
                        }
                        if (flag[b] == 1 && flag[d] == 1 &&
                            resv_dominant_old[d].fitness1 <= resv_dominant_old[b].fitness1 &&
                            resv_dominant_old[d].fitness2 <= resv_dominant_old[b].fitness2 &&
                            (resv_dominant_old[d].fitness1 < resv_dominant_old[b].fitness1 ||
                             resv_dominant_old[d].fitness2 < resv_dominant_old[b].fitness2)) {
                            flag[b] = 0;
                            dominant_number_all--;
                        }
                    }
                }

                int p = 0;
//                resv_dominant = (ant *) malloc((dominant_number_all) * sizeof(ant));
                for (int b = 0; b < count; b++) {
                    if (flag[b] == 1) {
                        for (int i = 0; i < order; i++)
                            resv_dominant[p].position[i] = resv_dominant_old[b].position[i];
                        resv_dominant[p].fitness1 = resv_dominant_old[b].fitness1;
                        resv_dominant[p].fitness2 = resv_dominant_old[b].fitness2;
                        resv_dominant[p].isDominant = 1;
                        p++;
                    }
                }
//                printf("dominant_number_all_zhu=%d\n",dominant_number_all);
//
//                Output
                if (iter < (max_iter - 1)) {
                    for (int i = 0; i < dominant_number_all; i++) {
                        cout<<snp_names[resv_dominant[i].position[0]]<<","<<snp_names[resv_dominant[i].position[1]]<<",f1="
//                        <<snp_names[resv_dominant[i].position[2]]<<","
                        <<resv_dominant[i].fitness1<<",f2="<<resv_dominant[i].fitness2<<",isDominant="<<
                               resv_dominant[i].isDominant<<endl;
                    }
                }

                if (iter == max_iter - 1) {
                    int xx = 0;
                    for (int i = 0; i < dominant_number_all; i++) {
                        if ((resv_dominant[i].position[0] == obj1 && resv_dominant[i].position[1] == obj2) ||
                            (resv_dominant[i].position[0] == obj2 && resv_dominant[i].position[1] == obj1)) {
                            xx = 1;
                        }
                        cout << snp_names[resv_dominant[i].position[0]] << ","
                             << snp_names[resv_dominant[i].position[1]] << ","
//                             << snp_names[resv_dominant[i].position[2]] << ","
                             << resv_dominant[i].fitness1 << ","
                             << resv_dominant[i].fitness2 << "," <<
                             resv_dominant[i].isDominant << endl;
                    }
                    if (xx == 1)
                        Right_Dataset_Num = Right_Dataset_Num + 1;
                    printf("---------------------------------Right_Dataset_Num=%d\n", Right_Dataset_Num);
                }


            }
            //endregion

            //region crossover
            if (rank == size - 1) {
                int rd;
                int cc = 0;
                int dd = 0;
                mate_number = min(10, dominant_number_all * (dominant_number_all - 1) / 2);
//                resv_transfer_mate = (ant *) malloc((mate_number) * sizeof(ant));
                for (int i = 0; i < dominant_number_all; i++) {
                    for (int j = 0; j < dominant_number_all; j++) {
                        if (i < j) {
                            rd = (rand() % (order - 1)) + 1; //选择交叉，单点交叉
                            for (int a = 0; a < rd; a++) {
                                resv_transfer_mate[cc].position[a] = resv_dominant[i].position[a];
                            }
                            for (int a = rd; a < order; a++) {
                                resv_transfer_mate[cc].position[a] = resv_dominant[j].position[a];

                                //De-duplication----------------------------
                                for (int g = 0; g < rd; g++) {
                                    if (resv_transfer_mate[cc].position[a] == resv_transfer_mate[cc].position[g]) {
                                        resv_transfer_mate[cc].position[a] = resv_dominant[j].position[g];
                                    }
                                }
                            }
                            cc = cc + 1;
                            if (cc == mate_number) {
                                dd = 1;
                                break;
                            }
                        }
                    }
                    if (dd = 1)
                        break;
                }
            }
            //endregion

            //endregion End of information transfer module


//            region 4.6.2 Feature Regrouping Module
            if (rank != size - 1) {
                MPI_Barrier(comm1);
                MPI_Gather(local_exchange_array, local_coll_num, MPI_INT, recv_snp, local_coll_num, MPI_INT, 0, comm1);
                int send[local_coll_num * (size - 1)];
                if (rank == 0) {
                    Regroup(recv_snp, send, size - 1, local_coll_num * (size - 1));
                }
                MPI_Scatter(send, local_coll_num, MPI_INT, local_exchange_array, local_coll_num, MPI_INT, 0, comm1);
                if (rank < size - 1) {
                    for (int k = 0; k < local_coll_num; k++) {
                        select_range[k] = local_exchange_array[k];
                    }
                }
            }


//            if (rank != size - 1) {
//                MPI_Barrier(comm1);
//                if (rank == 0) {
//                    std::vector<int> v;
//                    int send[snp_number];
//                    for (int a = 0; a < snp_number; a++) {
//                        v.push_back(a);
//                    }
//                    std::random_device rd;  
//                    std::mt19937 rnd(time(0));;  
//                    std::shuffle(v.begin(), v.end(), rnd);
//                    for(int a=0;a<snp_number;a++){
//                        send[a]=v[a];
//                    }
//                    MPI_Scatter(send, local_coll_num, MPI_INT, local_exchange_array, local_coll_num, MPI_INT, 0, comm1);
//                        for (int k = 0; k < local_snp_number; k++) {
//                            select_range[k] = local_exchange_array[k];
//                        }
//            }
//
//
//        }

        MPI_Barrier(MPI_COMM_WORLD);
        //endregion 

        if (rank == 0)
            printf("--------The %d-th outer loop ends--------\n",  iter);
    }
    //endregion Outer loop ends




    //Output Runtime
    end2 = clock();
    cost = (end2 - start2) / CLOCKS_PER_SEC;
    if (rank == 0)
        printf("Running_time= %f\n", cost);
    if (rank == 0 && !recv_snp)
        free(recv_snp);
    recv_snp = NULL;

    //region 4.6.3 Free memory

    if (!dominant) {
        free(dominant);
        dominant = NULL;
    }
    if (rank == size - 1) {
        if (!resv_dominant_old) {
            free(resv_dominant_old);
            resv_dominant_old = NULL;
        }
        if (!resv_dominant) {
            free(resv_dominant);
            resv_dominant = NULL;
        }
        if (!resv_transfer_mate) {
            free(resv_transfer_mate);
            resv_transfer_mate = NULL;
        }
    }

    cudaFree(d_data);
    cudaFree(dc);
    cudaFree(d_label);


    //endregion 


}
//endregion 

free(example);
free(displs);
free(recvCount);

MPI_Finalize();


return 0;


}


