/*************************************************************************************************
 * File: matrixmath.cu
 * Date: 11/06/2018
 * 
 * Compiling: Requires a Nvidia CUDA capable graphics card and the Nvidia GPU Computing Toolkit.
 *            Linux & Windows: nvcc -Wno-deprecated-gpu-targets -O3 -o prog2 matrixmath.cu
 *          
 * Usage:   Linux: >> prog2
 *          Windows: PS > ./prog2.exe
 * 
 * Description: This file runs a parallel program using CUDA to find the sum of squares. The first 
 *      part of the program asks whether you would like to run the optimized completely parallel 
 *      solution or an equivalent sequential solution. Both solutions use CUDA, but 1 is optimized 
 *      to be ran on many cores using atomic addition while the other runs the entire calculation 
 *      on a single pass-through, similar to how a sequential CPU program would run. Once the type 
 *      of kernel to run has been decided the user is asked how large they would like the sum of 
 *      squares to calculate. This calculation is done by creating an NxN matrix and a N sized vector.
 *      The matrix (A) and the vector (B) create a new vector C that satisfies the following formula:
 *      C[i] += A[i][j] * B[j]
 *
 *************************************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <fstream>

#define GRIDVAL 20.0

/**
 * __global__ void matrixSum(int, int, int, int, int)
 * - Function is a __global__ function meaning it is accessible for GPGPU processing.
 *   The function takes in a NxN matrix as *a, and a N length vector *b and an empty 
 *   N length vector *c along with the N value (both l and w are N in this case). The
 *   function calculates c[x] += a[x][y] * b[y] and performs an atomicAdd function when
 *   adding into c[x]. This function is meant to be highly parallelized.
 **/
__global__ void matrixSum(int *a, int *b, int *c, int l, int w) {
    // grab x position on grid
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    // grab y position on grid
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    // safety check + math run
    if( x >= 0 && y >= 0 && x < w && y < l) {
        // perform c[x] += a[y][x] * b[y] using an atomic add
        atomicAdd(&c[x], a[(x*w)+y] * b[y]);
    }
}

/**
 * __global__ void singleSum(int, int, int ,int ,int)
 * - Function is a __global__ function meaning it is accessible for GPGPU processing.
 *   The function takes in a NxN matrix as *a, and a N length vector *b and an empty
 *   N length vector *c along with the N value (both l and w are N in this case). The
 *   function loops through each y value and each x value calculating 
 *   c[x] += a[x][y] * b[y]. The function is meant to run on a single CUDA core and is
 *   meant to represent a sequential run of the matrixSum function
 **/
__global__ void singleSum(int *a, int *b, int *c, int l, int w) {
    // loop through all y values
    for(int i = 0; i < w; i++) {
        // loop through all x values
        for(int j = 0; j < l; j++) {
            // perform c[i] += a[y][x] * b[x]
            c[i] += a[(i*w)+j]*b[j];
        }
    }
}

/**
 * int main(int, char*[])
 * - Function is the entry point for the program. Welcomes the user, then asks the user 
 *   whether they want to run a sequential or parallel calculation for the sum of squares.
 *   Once a selection is made the program asks the user for the max square to use (also 
 *   known as the size N for the NxN matrix and N length vectors). When both these values
 *   have been entered then the NxN matrix and N length vectors are allocated and initialized
 *   with their starting values, the function then calls the external __global__ function 
 *   with the appropriate grid/block set-up and returns the result out.
 */
int main(int argc, char* argv[]) {

    // declare a size variable and a sequential flag
    int size, sequential;

    // give a hello prompt and prompt for either sequential or parallel
    std::cout << "Sum of Squares using CUDA." << std::endl;
    std::cout << "Enter 1 for Sequential calculation or enter 0 for Parallel calculation: ";
    std::cin >> sequential;
    
    // let the user know the selection they just made
    if (sequential == 1) 
        std::cout << "SEQUENTIAL calculation is ON." << std::endl << std::endl;
    else 
        std::cout << "PARALLEL calculation is ON." << std::endl << std::endl;
    
    // prompt user for N value of the matrix and vector
    std::cout << "Enter in the maximum square to calculate: ";
    std::cin >> size;

    // prepare a NxN matrix, and two N length vectors and populate them with valid data
    int *a = new int[size*size];
    int *b = new int[size];
    int *c = new int[size];
    for(int i = 0; i < size; i++) {
        for(int j = 0; j < size; j++) {
            a[(i*size)+j] = j+1;
        }
        b[i] = i+1;
        c[i] = 0;
    }

    // declare 3 variables that will be used on the GPU
    int *gpu_a, *gpu_b, *gpu_c;

    // allocate space on the GPU for the incoming matrix and vectors
    cudaMalloc( (void**)&gpu_a, (size * size)*sizeof(int));
    cudaMalloc( (void**)&gpu_b, (size)*sizeof(int));
    cudaMalloc( (void**)&gpu_c, (size)*sizeof(int));

    // copy all the matrix and vector data to the GPU, set gpu_c to be all 0s
    cudaMemcpy(gpu_a, a, size*size*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_b, b, size*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(gpu_c, 0, size*sizeof(int));

    // create a dim3 go find the number of blocks and number of threads per block given the user's input size
    // and the staticly defined GRIDVAL variable
    dim3 threadsPerBlock(GRIDVAL, GRIDVAL, 1);
    dim3 numBlocks(ceil(size/GRIDVAL), ceil(size/GRIDVAL), 1);

    // if we are running the sequential program, run the singleSum function with 1 block and 1 thread
    if (sequential == 1) 
        singleSum<<<1, 1>>>(gpu_a, gpu_b, gpu_c, size, size);
    // if w are running the parallel program, run the matrixSum function with the previously calculated num of blocks & threads
    else
        matrixSum<<<numBlocks, threadsPerBlock>>>(gpu_a, gpu_b, gpu_c, size, size);
    
    // copy the results from the GPGPU computation back to the CPU
    cudaMemcpy(c, gpu_c, size*sizeof(int), cudaMemcpyDeviceToHost);

    // output the result of C
    printf("Resulting values of the vector C:\n");
    for(int i = 0; i < size; i++) {
        printf("%d | ", c[i]);
    }
    printf("\n");

    //return a 0 for successful program run.
    return 0;
}
