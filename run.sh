#!/bin/bash
echo "Cleaning up old files..."
rm -f *.o *.out

echo "Compiling main.cpp..."
mpic++ -c ./main.cpp -I/usr/local/cuda/include -L/usr/local/cuda/lib64

if [ $? -ne 0 ]; then
	echo "Error: Failed to compile main.cpp"
	exit 1
fi

echo "Compiling CUDA files (utils.cu and EA_GPU.cu)..."
nvcc -c ./utils.cu EA_GPU.cu -std=c++11

if [ $? -ne 0 ]; then
	echo "Error: Failed to compile CUDA files"
	exit 1
fi

echo "Linking object files..."
mpic++ EA_GPU.o utils.o main.o -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart

if [ $? -ne 0 ]; then
	echo "Error: Failed to link object files"
	exit 1
fi

#Assuming the parallelism is 5.
mpirun --allow-run-as-root -np 5 ./a.out
