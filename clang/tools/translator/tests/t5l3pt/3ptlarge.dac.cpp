#include <any>
#include <iostream>
#include <vector>

#include "ReconTensor.h"

namespace dacpp {
typedef std::vector<std::any> list;
}

shell dacpp::list PDE3(const dacpp::Vector<float>& u_kin,
                       dacpp::Vector<float>& u_kout,
                       const dacpp::Vector<float>& r) {
    dacpp::index i;
    dacpp::split s(3, 1);
    binding(i, s);
    dacpp::list dataList{u_kin[s], u_kout[i], r[{}]};
    return dataList;
}

calc void pde3(dacpp::Vector<float>& u_kin,
               float* u_kout,
               float* r) {
    u_kout[0] = r[0] * u_kin[0] +
                (1.0f - 2.0f * r[0]) * u_kin[1] +
                r[0] * u_kin[2];
}

int main() {
    constexpr int input_size = 41943040 + 2;
    std::vector<float> u_data(input_size, 0.0f);
    for (int i = 0; i < input_size; ++i) {
        u_data[i] = static_cast<float>(i % 65536);
    }
    std::vector<float> out_data(input_size - 2, 0.0f);
    std::vector<float> r_data{0.25f};

    dacpp::Vector<float> u_tensor(u_data);
    dacpp::Vector<float> out_tensor(out_data);
    dacpp::Vector<float> r_tensor(r_data);

    PDE3(u_tensor, out_tensor, r_tensor) <-> pde3;
    out_tensor.print();
    return 0;
}
