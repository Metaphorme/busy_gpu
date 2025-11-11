# Makefile for Busy GPU

# Version information
VERSION := 1.0.0
BUILD_DATE := $(shell date +%Y-%m-%d)
BUILD_OS := $(shell uname -s)
BUILD_ARCH := $(shell uname -m)

# Compiler
NVCC := nvcc

# Target executable
TARGET := busy_gpu

# Source files
SOURCES := busy_gpu.cu

# GPU architectures (SM versions)
# SM 5.2: Maxwell (GTX 9xx, Titan X)
# SM 6.0: Pascal (P100)
# SM 6.1: Pascal (GTX 10xx, Titan Xp)
# SM 7.0: Volta (V100)
# SM 7.5: Turing (RTX 20xx, Titan RTX)
# SM 8.0: Ampere (A100)
# SM 8.6: Ampere (RTX 30xx, A40)
# SM 8.9: Ada Lovelace (RTX 40xx, L40)
# SM 9.0: Hopper (H100)
GPU_ARCHS_LIST := 52 60 61 70 75 80 86 89 90

# Generate GPU_ARCHS_STR for version info (comma-separated)
GPU_ARCHS_STR := "$(subst $(space),$(comma),$(GPU_ARCHS_LIST))"

# Generate gencode flags from GPU_ARCHS_LIST
comma := ,
space := $(empty) $(empty)
GPU_ARCHS := $(foreach arch,$(GPU_ARCHS_LIST),-gencode arch=compute_$(arch)$(comma)code=sm_$(arch))

# Compiler flags
NVCC_FLAGS := -O3 -use_fast_math -cudart static \
	-DVERSION=\"$(VERSION)\" \
	-DBUILD_DATE=\"$(BUILD_DATE)\" \
	-DBUILD_OS=\"$(BUILD_OS)\" \
	-DBUILD_ARCH=\"$(BUILD_ARCH)\" \
	-DGPU_ARCHS=\"$(GPU_ARCHS_STR)\"

# Default target
all: $(TARGET)

# Build target
$(TARGET): $(SOURCES)
	$(NVCC) $(NVCC_FLAGS) $(GPU_ARCHS) $< -o $@

# Clean build artifacts
clean:
	rm -f $(TARGET)

# Rebuild
rebuild: clean all

# Install (copy to /usr/local/bin, requires sudo)
install: $(TARGET)
	install -m 755 $(TARGET) /usr/local/bin/

# Uninstall (remove /usr/local/bin/busy_gpu, requires sudo)
uninstall:
	rm -f /usr/local/bin/$(TARGET)

# Run the program
run: $(TARGET)
	./$(TARGET)

# Show GPU info
info:
	nvidia-smi

# Help
help:
	@echo "Busy GPU Makefile"
	@echo ""
	@echo "Usage:"
	@echo "  make          - Build the program (default)"
	@echo "  make clean    - Remove built files"
	@echo "  make rebuild  - Clean and rebuild"
	@echo "  make run      - Build and run the program"
	@echo "  make install  - Install to /usr/local/bin (requires sudo)"
	@echo "  make uninstall- Uninstall from /usr/local/bin (requires sudo)"
	@echo "  make info     - Show GPU information"
	@echo "  make help     - Show this help message"
	@echo ""

.PHONY: all clean rebuild install uninstall run info help
