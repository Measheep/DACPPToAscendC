#include <any>
#include <iostream>
#include <vector>

#include "ReconTensor.h"

namespace dacpp {
typedef std::vector<std::any> list;
}

const int IN_FEATURES = 64;
const int OUT_FEATURES = 1048576;

shell dacpp::list linear_shell(const dacpp::Vector<float>& input,
                               const dacpp::Matrix<float>& weight,
                               const dacpp::Vector<float>& bias,
                               dacpp::Vector<float>& output) {
    dacpp::index out;
    dacpp::list dataList{input[{}], weight[out][{}], bias[out], output[out]};
    return dataList;
}

calc void linear_calc(dacpp::Vector<float>& input,
                      dacpp::Vector<float>& weight_row,
                      float* bias,
                      float* output) {
    float sum = bias[0];
    for (int i = 0; i < IN_FEATURES; ++i) {
        sum += input[i] * weight_row[i];
    }
    output[0] = sum;
}

int main() {
    std::vector<float> input_data(IN_FEATURES, 0.0f);
    for (int i = 0; i < IN_FEATURES; ++i) {
        input_data[i] = static_cast<float>(i) * 0.01f;
    }

    std::vector<float> weight_data(OUT_FEATURES * IN_FEATURES, 0.0f);
    for (int idx = 0; idx < OUT_FEATURES * IN_FEATURES; ++idx) {
        weight_data[idx] = static_cast<float>(idx % IN_FEATURES) * 0.001f;
    }

    std::vector<float> bias_data(OUT_FEATURES, 0.1f);
    std::vector<float> output_data(OUT_FEATURES, 0.0f);

    dacpp::Vector<float> input_tensor(input_data);
    dacpp::Matrix<float> weight_tensor({OUT_FEATURES, IN_FEATURES}, weight_data);
    dacpp::Vector<float> bias_tensor(bias_data);
    dacpp::Vector<float> output_tensor(output_data);

    linear_shell(input_tensor, weight_tensor, bias_tensor, output_tensor) <-> linear_calc;
    output_tensor.print();
    return 0;
}
