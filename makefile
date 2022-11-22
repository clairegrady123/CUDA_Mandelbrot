COMPILER = nvcc
CFLAGS = -I /usr/local/cuda-11.4/samples/common/inc
COBJS = bmpfile.o
EXES = cgrady3_mandelbrot

all: ${EXES}

cgrady3_mandelbrot: cgrady3_mandelbrot.cu ${COBJS}
	${COMPILER} ${CFLAGS} cgrady3_mandelbrot.cu ${COBJS} -o cgrady3_mandelbrot -lm


%.o: %.c %.h  makefile
	${COMPILER} ${CFLAGS} -lm $< -c

clean:
	rm -f *.o *~ ${EXES}

run:
	./cgrady3_mandelbrot 1080 1920
