NVCC_FLAGS = -std=c++17 -O3 -w
NVCC_LDFLAGS = -lcublas -lcuda 
# OUT_DIR = out

CUDA_OUTPUT_FILE = -o $@
# NCU_PATH := $(shell which ncu)
# NCU_COMMAND = sudo $(NCU_PATH) --set full --import-source yes

NVCC_FLAGS += --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math -Xcompiler=-fPIE -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing
NVCC_FLAGS += -arch=sm_90a

NVCC_BASE = nvcc $(NVCC_FLAGS) $(NVCC_LDFLAGS) -lineinfo

topk: topk.cu 
	$(NVCC_BASE) $^ $(CUDA_OUTPUT_FILE)

clean:
	rm topk