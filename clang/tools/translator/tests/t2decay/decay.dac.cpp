#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include "ReconTensor.h"

namespace dacpp {
    typedef std::vector<std::any> list;
}

const float dt = 0.1;       // 时间步长
const float T = 40.0;      // 总时间
const size_t numIsotopes = 524288; // 同位素数量
shell dacpp::list DECAY(const dacpp::Vector<float>& N0s,
                        const dacpp::Vector<float>& lambdas,
                        dacpp::Vector<float>& local_A,
                        const dacpp::Vector<float>& t) {
    dacpp::index i;
    dacpp::list dataList{N0s[i], lambdas[i], local_A[i], t[{}]};
    return dataList;
}

calc void decay(float* N0s,
                float* lambdas,
                float* local_A,
                float* t) {
    local_A[0] = N0s[0] * std::exp(-lambdas[0] * t[0]); 
}

// 计算每种同位素在时间 t 的数量
void calculateDecay(const std::vector<float>& lambdas, const std::vector<float>& N0s, float dt, float T) {
    size_t numIsotopes = lambdas.size(); // 同位素的数量
    std::vector<float> A(T/dt*numIsotopes, 0.0);  // 存储每个同位素在不同时间点的数量
    std::vector<float> time;  // 时间序列
    std::vector<float> t;
    t.push_back(static_cast<float>(0));

    // 串行计算每个同位素的衰变过程
    std::vector<float> local_A(numIsotopes, 0.0);
    dacpp::Vector<float> local_A_tensor(local_A);
    dacpp::Vector<float> N0s_tensor(N0s);
    dacpp::Vector<float> lambdas_tensor(lambdas);
    dacpp::Vector<float> t_tensor(t);
    dacpp::Matrix<float> A_tensor({static_cast<int>(T/dt), static_cast<int>(numIsotopes)}, A);
    

    while(t_tensor[0] <= T){  
        DECAY(N0s_tensor, lambdas_tensor, local_A_tensor,  t_tensor) <-> decay;
        A_tensor[10*t_tensor[0]] = local_A_tensor;
        t_tensor[0] += dt;
    }
    A_tensor[1].print();
}

int main() {
    

    // 随机生成衰变常数和初始数量
    std::vector<float> lambdas(numIsotopes);
    std::vector<float> N0s(numIsotopes, 1000.0);  // 初始数量为1000

    // 初始化衰变常数，与 host.cpp 的 LambdaAt 保持一致
    for (size_t i = 0; i < numIsotopes; ++i) {
        lambdas[i] = 0.01f + 0.0001f * static_cast<float>(i % 1024);
    }
    calculateDecay(lambdas, N0s, dt, T);

    return 0;
}
