#include <iostream>
#include <vector>
#include "ReconTensor.h"

namespace dacpp {
    typedef std::vector<std::any> list;
}

using namespace std;

shell dacpp::list matrixMultiply_shell(const dacpp::Matrix<float>& matA,
                                        const dacpp::Matrix<float>& matB,
                                        dacpp::Matrix<float>& matC) {
    dacpp::index idx1, idx2;
    dacpp::list dataList{matA[idx1][{}], matB[{}][idx2], matC[idx1][idx2]};
    return dataList;
}

calc void matrixMultiply_calc(dacpp::Vector<float>& vecA,
                dacpp::Vector<float>& vecB,
                float* dotProduct) {
    for (int i = 0; i < 5; i++) {
        dotProduct[0] += vecA[i] * vecB[i];
    }
}

int main() {
    // 初始化两个矩阵 A 和 B
    std::vector<float> dataA(2048*4096, 1.0f); // A 矩阵元素初始化为 1.0
    dacpp::Matrix<float> matA({2048, 4096}, dataA);

    std::vector<float> dataB(4096*2048, 1.0f); // B 矩阵元素初始化为 1.0
    dacpp::Matrix<float> matB({4096, 2048}, dataB);
    dacpp::Matrix<float> matC({2048, 2048}, 0);

    matrixMultiply_shell(matA, matB, matC) <-> matrixMultiply_calc;
    matC.print();

    return 0;
}
