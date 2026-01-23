# Makefile for FHT x86 optimization benchmarks
# Used by gen_x86.py to measure performance of different optimization strategies

CXX ?= g++
BENCHMARK_INCLUDE ?= /usr/local/include
BENCHMARK_LIB ?= /usr/local/lib

CXX_FLAGS = -O3 -Wall -march=native -std=c++11 \
            -I$(BENCHMARK_INCLUDE) \
            -L$(BENCHMARK_LIB) \
            -Wl,-rpath,$(BENCHMARK_LIB) \
            -lbenchmark -lpthread

.PHONY: run_float run_double clean

run_float: run_float.cpp to_run.h
	$(CXX) run_float.cpp -o run_float $(CXX_FLAGS)

run_double: run_double.cpp to_run.h
	$(CXX) run_double.cpp -o run_double $(CXX_FLAGS)

clean:
	rm -rf run_float run_double to_run.h output.csv
