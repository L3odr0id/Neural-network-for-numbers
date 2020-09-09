#ifndef TIMER_H
#define TIMER_H

#include <chrono>
#include <stdio.h>
#include <iostream>
#include <iomanip>

using namespace std;

struct Clock
{
    Clock (const char* string){
        start_time = chrono::high_resolution_clock::now();
        label = string;
    }

    void get_info(){
         chrono::duration<float> seconds = chrono::high_resolution_clock::now() - start_time;
         cout<<label;
         cout<<fixed<<setprecision(2)<<seconds.count()<<" seconds"<<endl;
    }

    ~Clock (){}

    chrono::high_resolution_clock::time_point start_time;
    const char* label;
};

#endif // TIMER_H
