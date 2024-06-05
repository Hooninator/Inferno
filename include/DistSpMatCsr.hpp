
#ifndef SPMATCSR_HPP
#define SPMATCSR_HPP

#include "common.h"
#include "utils.hpp"
#include "ProcessCube.hpp"

namespace inferno{

template <typename I, typename N, typename Device=upcxx::gpu_default_device>
class DistSpMatCsr 
{
public:


    template <typename T>
    using dev_ptr = upcxx::global_ptr<T, upcxx::memory_kind::cuda_device> ;

    template <typename T>
    using dist_dev_ptr = upcxx::dist_object<upcxx::global_ptr<T, upcxx::memory_kind::cuda_device>> ;

    template <typename T>
    using host_ptr = upcxx::global_ptr<T> ;

    template <typename T>
    using dist_host_ptr = upcxx::dist_object<upcxx::global_ptr<T>> ;

    DistSpMatCsr(int proc_rows, int proc_cols, int proc_layers)
    {

        /* Process cube setup */
        assert(proc_rows==proc_cols);
        assert(is_pow_2(proc_rows*proc_cols));

        proc_cube.reset(new ProcessCube(proc_rows, proc_cols, proc_layers));

        int rank = upcxx::rank_me();
        int world_size = upcxx::rank_n();

        assert(world_size == proc_cube->total_procs());


          
    }


    void read_mm(std::string& path)
    {
        /* TODO */
    }


    void local_spgemm() {/* TODO */}

    
    upcxx::future<> get_remote_tile()
    {
        /* TODO */
    }


private:

    // Process cube 
    std::shared_ptr<ProcessCube> proc_cube;
    

    // distributed objects containing pointers to csr arrays
    dist_dev_ptr<N> vals;
    dist_dev_ptr<I> rowinds;
    dist_dev_ptr<I> colptrs;


};


}//inferno

#endif
