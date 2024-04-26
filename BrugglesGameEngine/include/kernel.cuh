#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void VectorAdd(int* a, int* b, int* c);

void WrapperVectorAdd();