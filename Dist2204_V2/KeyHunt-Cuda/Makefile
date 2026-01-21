# Makefile for KeyHunt-Cuda
# Original version - setup script will modify for Ubuntu 22.04/24.04

CXX        = g++-8
CUDA       = /usr/local/cuda
CXXCUDA    = /usr/bin/g++-8
NVCC       = $(CUDA)/bin/nvcc

CXXFLAGS   = -m64 -mssse3 -Wno-write-strings -O2 -I. -I$(CUDA)/include
LFLAGS     = -lgmp -lpthread

OBJDIR     = obj

OBJS = $(OBJDIR)/Base58.o \
       $(OBJDIR)/IntGroup.o \
       $(OBJDIR)/Main.o \
       $(OBJDIR)/Bloom.o \
       $(OBJDIR)/Random.o \
       $(OBJDIR)/Timer.o \
       $(OBJDIR)/Int.o \
       $(OBJDIR)/IntMod.o \
       $(OBJDIR)/Point.o \
       $(OBJDIR)/SECP256K1.o \
       $(OBJDIR)/KeyHunt.o \
       $(OBJDIR)/GPU/GPUGenerate.o \
       $(OBJDIR)/hash/ripemd160.o \
       $(OBJDIR)/hash/sha256.o \
       $(OBJDIR)/hash/sha512.o \
       $(OBJDIR)/hash/ripemd160_sse.o \
       $(OBJDIR)/hash/sha256_sse.o \
       $(OBJDIR)/hash/keccak160.o \
       $(OBJDIR)/GmpUtil.o \
       $(OBJDIR)/CmdParse.o

# CPU-only build by default
ccap = 75

ifdef gpu
    CXXFLAGS += -DWITHGPU
    LFLAGS += -L$(CUDA)/lib64 -lcudart
    OBJS += $(OBJDIR)/GPU/GPUEngine.o
endif

# Default target
all: KeyHunt

$(OBJDIR):
	mkdir -p $(OBJDIR)
	cd obj &&	mkdir -p GPU
	cd obj &&	mkdir -p hash

ifdef gpu
ifdef debug
$(OBJDIR)/GPU/GPUEngine.o: GPU/GPUEngine.cu
	$(NVCC) -G -maxrregcount=0 --ptxas-options=-v --compile --compiler-options -fPIC -ccbin $(CXXCUDA) -m64 -g -I$(CUDA)/include -gencode=arch=compute_$(ccap),code=sm_$(ccap) -o $(OBJDIR)/GPU/GPUEngine.o -c GPU/GPUEngine.cu
else
$(OBJDIR)/GPU/GPUEngine.o: GPU/GPUEngine.cu
	$(NVCC) -maxrregcount=0 --ptxas-options=-v --compile --compiler-options -fPIC -ccbin $(CXXCUDA) -m64 -O2 -I$(CUDA)/include -gencode=arch=compute_$(ccap),code=sm_$(ccap) -o $(OBJDIR)/GPU/GPUEngine.o -c GPU/GPUEngine.cu
endif
endif

$(OBJDIR)/%.o : %.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $<

all: KeyHunt

KeyHunt: $(OBJDIR) $(OBJS)
	@echo Making KeyHunt...
	$(CXX) $(OBJS) $(LFLAGS) -o KeyHunt

clean:
	@echo Cleaning...
	@rm -f $(OBJDIR)/*.o
	@rm -f $(OBJDIR)/GPU/*.o
	@rm -f $(OBJDIR)/hash/*.o
	@rm -f KeyHunt
