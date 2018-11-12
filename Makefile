################ MAKEFILE TEMPLATE ################

# Author : Lucas Carpenter

# Usage : make target1

# What compiler are we using? (gcc, g++, nvcc, etc)
LINK = nvcc

# Name of our binary executable
OUT_FILE = prog2

# Any weird flags ( -O2/-O3/-Wno-deprecated-gpu-targets/-fopenmp/etc)
FLAGS = -Wno-deprecated-gpu-targets -O3 -D_FORCE_INLINES -std=c++11

all: prog2

prog2: matrixmath.cu
	$(LINK) -o $(OUT_FILE) $(FLAGS) $^

clean: 
	rm -f *.o *~ core
