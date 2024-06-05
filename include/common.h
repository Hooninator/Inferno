#ifndef COMMON_H
#define COMMON_H


#include <vector>
#include <map>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <cassert>


#include <cuda.h>
#include <cublas.h>
#include <cusparse.h>

#include <upcxx/upcxx.hpp>


#define CUDA_CHECK(call) {                                                 \
    cudaError_t err = call;                                                \
    if (err != cudaSuccess) {                                              \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " ("     \
                  << __FILE__ << ":" << __LINE__ << ")" << std::endl;      \
        exit(err);                                                         \
    }                                                                      \
}

namespace inferno {


class DeviceManager
{

public:

	using Device = upcxx::gpu_default_device;

    static upcxx::device_allocator<Device> ReserveAll()
    {
        size_t free, total;
        CUDA_CHECK(cudaMemGetInfo(&free, &total));

        auto allocator = upcxx::make_gpu_allocator<Device>(free);
        return allocator;
    }

    static void SetDevice()
    {
        CUDA_CHECK(cudaSetDevice(Device::auto_device_id));
    }

};


namespace globals {

	using Device = upcxx::gpu_default_device;

	upcxx::device_allocator<Device> allocator;

}

void inferno_init()
{
    upcxx::init();

	/* GPU setup */

	DeviceManager::SetDevice();
	globals::allocator = DeviceManager::ReserveAll();
	
}


void inferno_finalize()
{

	globals::allocator.destroy();

    upcxx::finalize();

}

}

#endif
