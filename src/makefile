SRCS := $(wildcard *.c)
OBJS := $(patsubst %.c, %.o, $(SRCS))

CUDA_SRCS := $(wildcard *.cu)
CUDA_OBJS := $(patsubst %.cu, %.o, $(CUDA_SRCS))

#OBJS = driver.o get_forces.o init_sphere.o jacobi_eigenvalue.o lbe.o main.o msg_ser.o n_list.o output.o penetration.o stretch.o update.o utils.o verlet_update.o
#CUDA_OBJS = lb.o coupling.o cuda_common_cuda.o

CUDA_PATH := /usr/local/cuda
NVCC := nvcc
GPU_CARD := -arch=sm_35
NVCC_FLAGS += -O3 -dc
NVCC_FLAGS += -Xcompiler -fopenmp
CUDA_LINK_FLAGS := -dlink
CUDA_LIB := -L${CUDA_PATH}/lib64 -lcudart #-lcublas -lcufft
CUDA_INC += -I$(CUDA_PATH)/include
CUDA_LINK_OBJ := cuLink.o
EXEC := dplbe # specify your executable name

CC := icc
CFLAGS += -std=gnu11 -O3 -Wall -ipo -qopenmp -xHost
INC += -I${MKLROOT}/include
INC += -I$(CUDA_PATH)/include
LIB := -L${MKLROOT}/lib/intel64 -lmkl_rt -lpthread -lm -ldl

LD := icc -qopenmp

all: $(EXEC)
$(EXEC): $(OBJS) $(CUDA_OBJS)
	$(NVCC) $(GPU_CARD) $(CUDA_LINK_FLAGS) $(CUDA_OBJS) -o $(CUDA_LINK_OBJ)
	$(LD) $(OBJS) $(CUDA_LINK_OBJ) $(CUDA_OBJS) -o $@ $(CUDA_LIB)

#$(EXEC): $(OBJS) $(CUDA_OBJS)
#	$(NVCC) $(GPU_CARD) $(NVCC_FLAGS) $(CUDA_OBJS) $(OBJS) -o $@

%.o: %.cu
	${NVCC} -c $< -o $@ ${CUDA_INC} ${GPU_CARD} ${NVCC_FLAGS} 

%.o: %.c
	${CC} -c $< -o $@ ${INC} ${CFLAGS} ${LIB}



clean:
	rm -f $(OBJS) $(EXEC) *.o a.out


 




 
