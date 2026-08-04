#include <any>
#include <iostream>
#include <vector>
#include "ReconTensor.h"

namespace dacpp {
typedef std::vector<std::any> list;
}

const int WIDTH = 41943040;

void initializeInput(std::vector<float>& rho) {
    for (int i = 0; i <= WIDTH; ++i) {
        rho[i] = static_cast<float>(i % 65536);
    }
}

shell dacpp::list chafen_shell(const dacpp::Vector<float>& rho,
                               dacpp::Vector<float>& diff) {
    dacpp::index idx;
    dacpp::split S1(2, 1);
    binding(idx, S1);
    dacpp::list dataList{rho[S1], diff[idx]};
    return dataList;
}

calc void chafen_calc(dacpp::Vector<float>& rho, float* diff) {
    diff[0] = rho[1] - rho[0];
}

int main() {
    std::vector<float> rho(WIDTH + 1, 0.0f);
    std::vector<float> diff(WIDTH, 0.0f);
    initializeInput(rho);

    dacpp::Vector<float> rho_tensor(rho);
    dacpp::Vector<float> diff_tensor(diff);

    chafen_shell(rho_tensor, diff_tensor) <-> chafen_calc;

    diff_tensor.print();
    return 0;
}
