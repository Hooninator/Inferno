
#ifndef LOG_HPP
#define LOG_HPP

#include <iostream>
#include <string>
#include <fstream>


namespace inferno {


class Log
{
public:

    Log(int rank)
    {
        ofs.open("Logfile"+std::to_string(rank)+".out");
    }

    std::ofstream& OFS() {return ofs;}
private:
    std::ofstream ofs;

};



} //inferno

#endif
