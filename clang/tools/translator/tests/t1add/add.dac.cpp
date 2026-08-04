#include <iostream>
#include "ReconTensor.h"
namespace dacpp {
    typedef std::vector<std::any> list;
}
shell dacpp::list vecAdd_shell(const dacpp::Vector<int>& vecA, const dacpp::Vector<int>& vecB, dacpp::Vector<int>& vecC) {
    dacpp::index i;
    dacpp::list dataList{vecA[{i}], vecB[{i}], vecC[{i}]};
    return dataList;
}
calc void vecAdd_calc(int* a, int* b, int* c) {
    c[0] = a[0] + b[0];
}
int main() {
    std::vector<int> vecA(134348800, 1);
    std::vector<int> vecB(134348800, 1);
    std::vector<int> vecC(134348800, 0);
    dacpp::Vector<int> vecA_tensor(vecA);
    dacpp::Vector<int> vecB_tensor(vecB);
    dacpp::Vector<int> vecC_tensor(vecC);
    vecAdd_shell(vecA_tensor, vecB_tensor,vecC_tensor) <-> vecAdd_calc;
    vecC_tensor.print();
    return 0;
}