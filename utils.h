//
// Created by lft on 2022/11/4.
//

#ifndef DGEMT_ACO444_UTILS_H
#define DGEMT_ACO444_UTILS_H

#include <string>
#include <cuda.h>
#include "cuda_runtime.h"
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"
using namespace std;

int CountLines(string filename);
void g_read_data(string filename,int** &d_data,int* &d_label,int* &dc,int &sample_number,int &snp_number,vector <string> &snp_names, int &r1,int &r2);


#endif //DGEMT_ACO444_UTILS_H
