/******************************************************************************
 * Created by Claire Grady on 13 / 9 / 2022.
 * cgrady3_mandelbrot.cu
 * Program that computes the mandelbrot set in parallel and produces
 * a bitmap image of a size selected by the user
 *
 * Parameters:
 *    1. Height of the image
 *    2. Width of the image
 * Returns: 0 on Success, exits on Failure
 *
 * Build:
 *    nvcc -I /usr/local/cuda-11.4/samples/common/inc 
 *    cgrady3_mandelbrot.cu -o cgrady3_mandelbrot
 * Run Example:
 *    ./cgrady3_mandelbrot <Image Width> <Image Height>
 * ***************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include "bmpfile.h"
#include <math.h>

/*Mandelbrot values*/
#define RESOLUTION 8700.0
#define XCENTER -0.51
#define YCENTER 0.57
#define MAX_ITER 1000
#define MAX_THREADS 1024
#define MAX_BLOCKS 1024

/*Colour Values*/
#define COLOUR_DEPTH 255
#define COLOUR_MAX 240.0
#define GRADIENT_COLOUR_MAX 230.0

#define FILENAME "cgrady3_mandelbrot_fractal.bmp"

 /** 
   * Computes the color gradiant
   * color: the output vector 
   * x: the gradiant (beetween 0 and 360)
   * min and max: variation of the RGB channels (Move3D 0 -> 1)
   * Check wiki for more details on the colour science: en.wikipedia.org/wiki/HSL_and_HSV 
   */
__device__ void GroundColorMix(double* color, double x, double min, double max)
{
  /*
   * Red = 0
   * Green = 1
   * Blue = 2
   */
    double posSlope = (max-min)/60;
    double negSlope = (min-max)/60;

    if( x < 60 )
    {
        color[0] = max;
        color[1] = posSlope*x+min;
        color[2] = min;
        return;
    }
    else if ( x < 120 )
    {
        color[0] = negSlope*x+2.0*max+min;
        color[1] = max;
        color[2] = min;
        return;
    }
    else if ( x < 180  )
    {
        color[0] = min;
        color[1] = max;
        color[2] = posSlope*x-2.0*max+min;
        return;
    }
    else if ( x < 240  )
    {
        color[0] = min;
        color[1] = negSlope*x+4.0*max+min;
        color[2] = max;
        return;
    }
    else if ( x < 300 )
    {
        color[0] = posSlope*x-4.0*max+min;
        color[1] = min;
        color[2] = max;
        return;
    }
    else
    {
        color[0] = max;
        color[1] = min;
        color[2] = negSlope*x+6*max;
        return;
    }
}

/*****************************************************************************
 * Function: parse_args
 *
 * Function to parse command line arguments
 *
 * Parameters:
 *     1. The number of command line parameters
 *     2. Array of pointers to each command line parameter
 *     5. The width of the image
 *     6. The height of the image
 *
 * Returns: 0 on Success, -1 on Failure
 *
 ****************************************************************************/

int parse_args(int argc, char *argv[], int *height, int *width){
    if ((argc != 3) ||
    ((*height = atoi(argv[1])) <= 0) ||
    ((*width = atoi(argv[2])) <= 0)) 
    {
        printf("Usage: %s <height> <width>\n", argv[0]);
        return(-1);
    }
    return(0);
}

/*****************************************************************************
 * Function: mandelbrot
 *
 * Function to compute the mandelbrot set and select the required RGB values
 *
 * Parameters:
 *     1. The width of the image
 *     2. The height of the image
 *     3. Pointer to an array to store the red pixel values
 *     4. Pointer to an array to store the green pixel values
 *     5. Pointer to an array to store the blue pixel values
 *
 * Returns: 0 on Success
 *
 ****************************************************************************/

__global__ void mandelbrot(int width, int height, double *gpu_R, double *gpu_G, double *gpu_B)
{
  
    int id = blockIdx.x * blockDim.x + threadIdx.x;


  
        if (id < width*height){
            
            int col = id % width;
            int row = id / width;
            double xoffset = -(width-1)/2.0;
            double yoffset = (height-1)/2.0;
            
            // Determine where in the mandelbrot set, the pixel is referencing
            double x = XCENTER + (xoffset + row) / RESOLUTION;
            double y = YCENTER + (yoffset - col) / RESOLUTION;

            // Mandelbrot stuff

            double a = 0.0;
            double b = 0.0;
            double aold = 0.0;
            double bold = 0.0;
            double zmagsqr = 0.0;
            int iter = 0;
            double x_col;
            double color[3];
            // Check if the x,y coord are part of the mendelbrot set - refer to the algorithm
            while (iter < MAX_ITER && zmagsqr <= 4.0)
            {
                ++iter;
                a = (aold * aold) - (bold * bold) + x;
                b = 2.0 * aold * bold + y;

                zmagsqr = a * a + b * b;
                aold = a;
                bold = b;
            }
            /* Generate the colour of the pixel from the iter value */
            /* You can mess around with the colour settings to use different gradients */
            /* Colour currently maps from royal blue to red */
            x_col = (COLOUR_MAX - ((((float)iter / ((float)MAX_ITER) * GRADIENT_COLOUR_MAX))));
            GroundColorMix(color, x_col, 1, COLOUR_DEPTH);
            gpu_R[id] = color[0];
            gpu_G[id] = color[1];
            gpu_B[id] = color[2];
            //if (id > 6242000)
                //printf(" th id: %d", id);
        }
}

/*****************************************************************************
 * Function: main
 *
 * Main program that creates the initalises all the required variables and
 * allocates memory for the CPU and GPU arrays. It calls the kernel function
 * then copys the data back to the CPU. Lastly it uses the arrays computed
 * in the GPU to create the bmp file and then cleans up the memory.
 * Parameters:
 *     1. The number of command line parameters
 *     2. Array of pointers to each command line parameter
 *
 * Returns: 0 on Success, , Exits on Failure with an Error Message
 *
 ****************************************************************************/

int main(int argc, char **argv)
{
  int height, width;
  if (parse_args(argc, argv, &height, &width) < 0)
  {
      printf("There was a problem parsing the command line arguments!\n");
      exit(EXIT_FAILURE);
  }
  //initialise variables
  bmpfile_t *bmp;
  rgb_pixel_t pixel = {0, 0, 0, 0};
  bmp = bmp_create(width, height, 32);
  size_t size = width*height * sizeof(double);

  //allocate CPU memory
  double *cpu_R = (double *)malloc(size);
  if (cpu_R == NULL)
  {
    printf("There was a problem with the malloc call\n");
    exit(EXIT_FAILURE);
  }
  double *cpu_G = (double *)malloc(size);
  if (cpu_G == NULL)
  {
    printf("There was a problem with the malloc call\n");
    exit(EXIT_FAILURE);
  }
  double *cpu_B = (double *)malloc(size);
  if (cpu_B == NULL)
  {
    printf("There was a problem with the malloc call\n");
    exit(EXIT_FAILURE);
  }

  //allocate GPU memory
  cudaError_t err = cudaSuccess;
  double *gpu_R, *gpu_G, *gpu_B;
  err = cudaMalloc((void**)&gpu_R, size);
  if (err != cudaSuccess)
  {
     printf("Failed to allocate device red array (error code %s)!\n", cudaGetErrorString(err));
     exit(EXIT_FAILURE);
  }
  err = cudaMalloc((void**)&gpu_G, size);
  if (err != cudaSuccess)
  {
     printf("Failed to allocate device green array (error code %s)!\n", cudaGetErrorString(err));
     exit(EXIT_FAILURE);
  }
  err = cudaMalloc((void**)&gpu_B, size);
  if (err != cudaSuccess)
  {
     printf("Failed to allocate device blue array (error code %s)!\n", cudaGetErrorString(err));
     exit(EXIT_FAILURE);
  }

  cudaEvent_t start;
  cudaEventCreate(&start);
  cudaEvent_t stop;
  cudaEventCreate(&stop);
  cudaEventRecord(start, NULL);

  //call mandelbrot kernel
  mandelbrot<<<width*height/MAX_BLOCKS, MAX_THREADS>>>(width, height, gpu_R, gpu_G, gpu_B);

  //copy device arrays back to host
  err = cudaMemcpy(cpu_R, gpu_R, size, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess)
  {
     printf("Failed to copy device to host red array (error code %s)!\n", cudaGetErrorString(err));
     exit(EXIT_FAILURE);
  }
  err = cudaMemcpy(cpu_G, gpu_G, size, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess)
  {
     printf("Failed to copy device to host green array (error code %s)!\n", cudaGetErrorString(err));
     exit(EXIT_FAILURE);
  }
  err = cudaMemcpy(cpu_B, gpu_B, size, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess)
  {
     printf("Failed to copy device to host blue array (error code %s)!\n", cudaGetErrorString(err));
     exit(EXIT_FAILURE);
  }

  cudaEventRecord(stop, NULL);
  cudaEventSynchronize(stop);
  float msecTotal = 0.0f;
  cudaEventElapsedTime(&msecTotal, start, stop);
  printf("Elapsed Time: %f ", msecTotal);

  //assign pixel values and then set them
  for (int i = 0; i < width*height; i++)
  {
    int col = i % width;
    int row = i / width;
    pixel.red = cpu_R[i];
    pixel.green = cpu_G[i];
    pixel.blue = cpu_B[i];
    bmp_set_pixel(bmp, col, row, pixel);
  }

  //save bmp image
  bmp_save(bmp, FILENAME);

  //free host memory
  free(cpu_R);
  free(cpu_G);
  free(cpu_B);

  //free device memory
  cudaFree(gpu_R);
  cudaFree(gpu_G);
  cudaFree(gpu_B);
  
  //destroy bmp
  bmp_destroy(bmp);

  return 0;
}