#ifndef UTILS_HPP
#define UTILS_HPP
#include <cmath>

namespace inferno {

template <typename T>
bool is_pow_2(T num)
{
    return ( std::ceil(std::log2(num)) == std::floor(std::log2(num)) ) ;
}

};

#endif
