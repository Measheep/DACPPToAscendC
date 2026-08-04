#include <iostream>
#include <vector>
#include <iomanip>
#include <algorithm>
#include <fstream>
#include <any>
#include <queue>
#include "ReconTensor.h"

namespace dacpp {
    typedef std::vector<std::any> list;
}

const int WIDTH = 2097152;  // 路段长度
const float TIME_STEPS = 256;  // 时间步数
const float DELTA_T = 0.01; // 时间步长
const float DELTA_X = 1.0;  // 空间步长

// 流量函数，考虑密度对流量的影响
float q(float rho) {
    float V_max = 30; // 最大速度
    float rho_max = 50; // 最大密度
    return rho * V_max * (1 - rho / rho_max);
}

// 初始化密度，使用随机的分布
void initializeDensity(std::vector<float>& rho) {
    for (int i = 0; i < WIDTH; ++i) {
        if (i < WIDTH / 4) {
            rho[i] = 40; // 高密度区
        } else if (i < 3 * WIDTH / 4) {
            rho[i] = 20; // 中密度区
        } else {
            rho[i] = 10; // 低密度区
        }
    }
}

// 计算交通流量
calc void lwr(dacpp::Vector<float> & rho, float* new_rho) {
    new_rho[0] = rho[1] - (DELTA_T / DELTA_X) * (q(rho[1]) - q(rho[0]));
    new_rho[0] = std::max(0.0f, new_rho[0]);
    //new_rho += 0.1 * (rho[0] + new_rho - 2 * rho[1]);
}

shell dacpp::list LWR_shell(const dacpp::Vector<float> & rho, 
                            dacpp::Vector<float> & new_rho) {
    dacpp::index idx1;
    dacpp::split S1(2, 1);
    binding(idx1, S1);
    dacpp::list dataList{rho[S1], new_rho[idx1]};
    return dataList;
}

int main() {
    // 创建 Tensor 类型对象
    std::vector<float> rho(WIDTH, 0.0);
    std::vector<float> new_rho(WIDTH, 0.0);
    initializeDensity(rho);
    dacpp::Vector<float> rho_tensor(rho);
    dacpp::Vector<float> new_rho_tensor(new_rho);
    dacpp::Vector<float> middle_out_tensor = new_rho_tensor[{1,WIDTH-1}];
    dacpp::Vector<float> middle_in_tensor = rho_tensor[{0,WIDTH-1}];
    for (int t = 0; t < TIME_STEPS; ++t) {
        LWR_shell(middle_in_tensor, middle_out_tensor) <-> lwr;
        for (int i = 1; i <= WIDTH-2; i++) {
            middle_in_tensor[i] = middle_out_tensor[i-1];
        }
        
        middle_in_tensor[0] = middle_out_tensor[0]; // 左边界无车流
        //middle_in_tensor[99] = middle_out_tensor[97];

    }
    std::cout << middle_in_tensor[15] << std::endl;
    

    

    // 释放动态分配的内存

    return 0;
}
