//
//#include <iostream>
//#include <cstring>
//#include <fstream>
//#include <sstream>
//#include <vector>
//#include "utils.h"
//
//#define CHECK(res) if(res!=cudaSuccess){exit(-1);}
//using namespace std;
//
//int CountLines(string filename) {
//    ifstream ReadFile;
//    int n = 0;
//    string tmp;
//    ReadFile.open(filename.c_str(), ios::in);//ios::in 表示以只读的方式读取文件
//    if (ReadFile.fail()) {
//        return 0;
//    } else {
//        while (getline(ReadFile, tmp, '\n')) {
//            n++;
//        }
//        ReadFile.close();
//        return n;
//    }
//}
//
//
//
//void g_read_data(string filename,int** &d_data,int* &d_label,int* &dc,int &sample_number,int &snp_number,vector <string> &snp_names) {
//
//    sample_number = 0;
//    snp_number = 0;
//
//    ifstream file;
//    file.open(filename.c_str(), ios::in);
//    if (file.fail()) {
//        cout << "文件不存在." << endl;
//        file.close();
//    } else {
//        string firstline;
//        string tmp;
////        vector <string> snp_names;
//
//        //统计snp数
//        getline(file, firstline);
//        stringstream fl(firstline);
//        while (getline(fl, tmp, ',')) {
//            snp_names.push_back(tmp);
//            snp_number++;
//        }
//        snp_number--;
//
//        //统计sample数
//        sample_number = CountLines(filename) - 1;
//
//        //读取data和label
//        int *data = NULL;
//        data = (int *) malloc(sample_number * snp_number * sizeof(int));
//        int *label = NULL;
//        label = (int *) malloc(sample_number * sizeof(int));
//        string line;
//
//        int row = 0;
//        int col = 0;
//        while (getline(file, line)) {
//            stringstream ss(line);
//            string tmp;
//            col = 0;
//            while (getline(ss, tmp, ',')) {
//                if (col == snp_number) {
//                    label[row] = tmp[0]-'0';
//                } else {
//                    data[row * snp_number + col] = tmp[0]-'0';
//                    col++;
//                }
//            }
//            row++;
//        }
//        file.close(); //关闭文件
//
//        printf("snp_number=%d,sample_number=%d\n",snp_number,sample_number);
//
//        //将data传到gpu上
//        int **ha = NULL;
//        cudaError_t res;
//        int r;
//
//        //传data
//        res = cudaMalloc((void **) (&d_data), sample_number * sizeof(int *));
//        CHECK(res);
//        res = cudaMalloc((void **) (&dc), sample_number * snp_number * sizeof(int));
//        CHECK(res);
//        ha = (int **) malloc(sample_number * sizeof(int *));
//
//        for (r = 0; r < sample_number; r++) {
//            ha[r] = dc + r * snp_number;
//        }
//        res = cudaMemcpy((void *) (d_data), (void *) (ha), sample_number * sizeof(int *), cudaMemcpyHostToDevice);
//        CHECK(res);
//        res = cudaMemcpy((void *) (dc), (void *) (data), sample_number * snp_number * sizeof(int),
//                         cudaMemcpyHostToDevice);
//        CHECK(res);
//
//        //传label
//        res = cudaMalloc((void **) (&d_label), sample_number * sizeof(int));
//        CHECK(res);
//        res = cudaMemcpy((void *) (d_label), (void *) (label), sample_number*sizeof(int), cudaMemcpyHostToDevice);
//        CHECK(res);
//
//        free(ha);
//        free(data);
//        free(label);
//
//
//    }
//
//}
//



#include <iostream>
#include <cstring>
#include <fstream>
#include <sstream>
#include <vector>
#include "utils.h"
#include<stdlib.h>
#include<time.h>
#include <cstdlib> 


#define CHECK(res) if(res!=cudaSuccess){exit(-1);}
using namespace std;

int CountLines(string filename) {
    ifstream ReadFile;
    int n = 0;
    string tmp;
    ReadFile.open(filename.c_str(), ios::in);//ios::in 表示以只读的方式读取文件
    if (ReadFile.fail()) {
        return 0;
    } else {
        while (getline(ReadFile, tmp, '\n')) {
            n++;
        }
        ReadFile.close();
        return n;
    }
}



void g_read_data(string filename,int** &d_data,int* &d_label,int* &dc,int &sample_number,int &snp_number,vector <string> &snp_names, int &r1,int &r2) {

    sample_number = 0;
    snp_number = 0;

    ifstream file;
    file.open(filename.c_str(), ios::in);
    if (file.fail()) {
        cout << "The file does not exist." << endl;
        file.close();
        exit(1);
    } else {
        string firstline;
        string tmp0;

        //统计snp数
        getline(file, firstline);
        stringstream fl(firstline);
        while (getline(fl, tmp0, '\t')) {
            snp_names.push_back(tmp0);
            snp_number++;
        }
        snp_number--;

        //统计sample数
        sample_number = CountLines(filename) - 1;

        //读取data和label
        int *data = NULL;
        data = (int *) malloc(sample_number * snp_number * sizeof(int));
        int *label = NULL;
        label = (int *) malloc(sample_number * sizeof(int));
        string line;

        int row = 0;
        int col = 0;
        while (getline(file, line)) {
            stringstream ss(line);
            string tmp;
            col = 0;
            while (getline(ss, tmp, '\t')) {
                if (col == snp_number) {
                    label[row] = tmp[0]-'0';
                } else {
                    data[row * snp_number + col] = tmp[0]-'0';
                    col++;
                }
            }
            row++;
        }
        file.close(); //关闭文件

        //交换两列
        srand(time(nullptr));
        r1=rand()%snp_number;
        r2=rand()%snp_number;

        // 确保 r1 和 r2 不重复
        while (r1 == r2) {
            r2 = rand() % snp_number;
        }

        int tmp1,tmp2;
        string tmp3,tmp4;

        for(int b=0;b<sample_number;b++){
            tmp1=data[b*snp_number+snp_number-2];
            data[b*snp_number+snp_number-2]=data[b*snp_number+r1];
            data[b*snp_number+r1]=tmp1;

            tmp2=data[b*snp_number+snp_number-1];
            data[b*snp_number+snp_number-1]=data[b*snp_number+r2];
            data[b*snp_number+r2]=tmp2;
        }

        // printf("Before exchange：%s %s\n",snp_names[snp_number-2].c_str(),snp_names[snp_number-1].c_str());

        tmp3=snp_names[snp_number-2];
        snp_names[snp_number-2]=snp_names[r1];
        snp_names[r1]=tmp3;

        tmp4=snp_names[snp_number-1];
        snp_names[snp_number-1]=snp_names[r2];
        snp_names[r2]=tmp4;

        // printf("Before exchange：%s %s\n",snp_names[snp_number-2].c_str(),snp_names[snp_number-1].c_str());

        printf("snp_number=%d,sample_number=%d,obj1=%d,obj2=%d\n",snp_number,sample_number,r1,r2);

        //将data传到gpu上
        int **ha = NULL;
        cudaError_t res;
        int r;

        //传data
        res = cudaMalloc((void **) (&d_data), sample_number * sizeof(int *));
        CHECK(res);
        res = cudaMalloc((void **) (&dc), sample_number * snp_number * sizeof(int));
        CHECK(res);
        ha = (int **) malloc(sample_number * sizeof(int *));

        for (r = 0; r < sample_number; r++) {
            ha[r] = dc + r * snp_number;
        }
        res = cudaMemcpy((void *) (d_data), (void *) (ha), sample_number * sizeof(int *), cudaMemcpyHostToDevice);
        CHECK(res);
        res = cudaMemcpy((void *) (dc), (void *) (data), sample_number * snp_number * sizeof(int),
                         cudaMemcpyHostToDevice);
        CHECK(res);


        //传label
        res = cudaMalloc((void **) (&d_label), sample_number * sizeof(int));
        CHECK(res);
        res = cudaMemcpy((void *) (d_label), (void *) (label), sample_number*sizeof(int), cudaMemcpyHostToDevice);
        CHECK(res);

        free(ha);
        free(data);
        free(label);


    }

}


