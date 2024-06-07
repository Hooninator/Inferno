#ifndef UTILS_HPP
#define UTILS_HPP
#include <cmath>

#include "common.h"

namespace inferno {

template <typename T>
bool is_pow_2(T num)
{
    return ( std::ceil(std::log2(num)) == std::floor(std::log2(num)) ) ;
}


void print_rank(std::string& str)
{
    std::cout<<"Rank "<<globals::mpi_rank<<str<<std::endl;

}
}

#endif
