#pragma once    
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Macro for error handling
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

// Function prototype
inline void HandleError(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}
