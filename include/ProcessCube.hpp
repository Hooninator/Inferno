#ifndef PROCESS_CUBE_HPP
#define PROCESS_CUBE_HPP

#include "common.h"

namespace inferno {
class ProcessCube
{

public:

    ProcessCube(int rows, int cols, int layers):
        rows(rows), cols(cols), layers(layers),
        grid_size(rows*cols),
        total_procs(rows*cols*layers),
        grid_dim( std::sqrt(rows*cols) )
    {
    }


    inline int get_rows() const { return rows; }
    inline int get_cols() const { return cols; }
    inline int get_layers() const { return layers; }
    inline int get_grid_size() const { return grid_size; }
    inline int get_total_procs() const { return total_procs; }
    inline int get_grid_dim() const { return grid_dim; }
        
private:

    int rows;
    int cols;
    int layers;
    int grid_size;
    int total_procs;
    int grid_dim;

};


}

#endif
