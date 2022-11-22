cgrady3_mandelbrot
Author: Claire Grady 
Version: 1.0

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
 *    nvcc -I /usr/local/cuda-11.4/samples/common/inc cgrady3_mandelbrot.cu -o cgrady3_mandelbrot
 * Run Example:
 *    ./cgrady3_mandelbrot <Image Width> <Image Height>
 * ***************************************************************************/