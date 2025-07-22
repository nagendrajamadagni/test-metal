XCODE = xcrun
CXX = clang++
SRC_DIR = src
METAL_DIR = $(SRC_DIR)/metal
CPP_DIR = $(SRC_DIR)/cpp
BUILD_DIR = build
BIN_DIR = bin
INCLUDE_DIR = include
METAL_CPP_DIR = $(SRC_DIR)/metal-cpp
METAL_SRC = $(shell find $(METAL_DIR) -name '*.metal')
METAL_REL_SRC = $(patsubst $(METAL_DIR)/%, %, $(METAL_SRC))
CPP_SRC = $(shell find $(CPP_DIR) -name '*.cpp')
CPP_REL_SRC = $(patsubst $(CPP_DIR)/%, %, $(CPP_SRC))
AIR_FILES = $(patsubst %.metal,$(BUILD_DIR)/%.air,$(METAL_REL_SRC))
OBJ_FILES = $(patsubst %.cpp,$(BUILD_DIR)/%.o,$(CPP_REL_SRC))
BIN_FILE = matmul
METAL_AR = $(BUILD_DIR)/matmul_kernel.metalar
METAL_LIB = $(BUILD_DIR)/matmul_kernel.metallib
CXX_FLAGS = -std=c++17 -fno-objc-arc -g
LD_FLAGS = -framework Metal -framework Foundation

SDK = macosx

.PHONY: all create_build_dir create_bin_dir build_air metal_ar build_metal_lib build_obj build_bin clean run
all: build_bin

create_build_dir:
	mkdir -p $(BUILD_DIR)

create_bin_dir:
	mkdir -p $(BIN_DIR)

$(BUILD_DIR)/%.air: $(METAL_DIR)/%.metal | create_build_dir
	mkdir -p $(dir $@)
	$(XCODE) --sdk $(SDK)  metal -c $< -o $@

build_air: $(AIR_FILES)

metal_ar: build_air
	$(XCODE) --sdk $(SDK) metal-ar rc $(METAL_AR) $(AIR_FILES)

build_metal_lib: metal_ar
	$(XCODE) --sdk $(SDK) metallib $(METAL_AR) -o $(METAL_LIB)

$(BUILD_DIR)/%.o: $(CPP_DIR)/%.cpp | create_build_dir
	 mkdir -p $(dir $@)
	 $(CXX) $(CXX_FLAGS) -c -I$(INCLUDE_DIR) -I$(METAL_CPP_DIR) $< -o $@

build_obj: $(OBJ_FILES)

build_bin: build_obj build_metal_lib create_bin_dir
	$(CXX) $(LD_FLAGS) $(OBJ_FILES) -o $(BIN_DIR)/$(BIN_FILE)

clean:
	rm -f $(BUILD_DIR)/*.air
	rm -f $(BUILD_DIR)/*.metallib
	rm -f $(BUILD_DIR)/*.metalar
	rm -f $(BUILD_DIR)/*.o
	rm -rf $(BUILD_DIR)/*
	rm -f $(BIN_DIR)/*

run: build_bin
	./$(BIN_DIR)/$(BIN_FILE)


