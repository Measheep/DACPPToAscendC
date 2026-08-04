#include <iostream>
#include <vector>
#include "ReconTensor.h"

namespace dacpp {
    typedef std::vector<std::any> list;
}

// 定义图像类型，假设每个像素由 RGB 三个值组成
struct Pixel {
    int r, g, b;
    // 重载 << 操作符，使 Pixel 类型的对象可以被输出
    friend std::ostream& operator<<(std::ostream& os, const Pixel& pixel) {
        os << "(" << pixel.r << ", " << pixel.g << ", " << pixel.b << ")";
        return os;
    }
};


shell dacpp::list imageAdjustment(const dacpp::Matrix<Pixel>& image_tensor, dacpp::Matrix<Pixel>& image_tensor2) {
    dacpp::index idx1, idx2;
    //dacpp::split s(3,1);
    //binding(i, s);
    dacpp::list dataList{image_tensor[idx1][idx2], image_tensor2[idx1][idx2]};
    return dataList;
}

// 色彩调整操作：增加红色分量
calc void image_1(Pixel* image_tensor,
                Pixel* image_tensor2) {
    image_tensor2[0].r = std::min(255, image_tensor[0].r + 50);  // 增加红色分量，并确保不超过255
}

// 亮度增强操作：增加每个像素的 RGB 分量
calc void image_2(Pixel* image_tensor2,
                Pixel* image_tensor3) {
    int value = 30;
    image_tensor3[0].r = std::min(255, image_tensor2[0].r + value);  // 增加红色分量，限制在255以内
    image_tensor3[0].g = std::min(255, image_tensor2[0].g + value);  // 增加绿色分量
    image_tensor3[0].b = std::min(255, image_tensor2[0].b + value);  // 增加蓝色分量}

}



// 打印图像的前几个像素，作为调试
void print_image(const std::vector<std::vector<Pixel>>& image, int num_rows = 5, int num_cols = 5) {
    for (int i = 0; i < num_rows; ++i) {
        for (int j = 0; j < num_cols; ++j) {
            std::cout << "(" << image[i][j].r << "," << image[i][j].g << "," << image[i][j].b << ") ";
        }
        std::cout << std::endl;
    }
}

int main() {
    // 使用不同的输入/输出初值，验证部分写 record 的未写字段来自输入透传。

    const int width = 1920;
    const int height = 1080;
    std::vector<Pixel> image(height * width, {100, 90, 80});
    std::vector<Pixel> image2(height * width, {1, 2, 3});
    //std::vector<std::vector<Pixel>> image2(height, std::vector<Pixel>(width, {1, 2, 3}));

    // 打印初始图像
    std::cout << "Original Image:" << std::endl;
    //print_image(image);

    dacpp::Matrix<Pixel> image_tensor({height, width}, image);
    dacpp::Matrix<Pixel> image_tensor2({height, width}, image2);

    // 执行色彩调整操作
    imageAdjustment(image_tensor, image_tensor2) <-> image_1;
    std::cout << "\nImage After Color Adjustment:" << std::endl;

    std::vector<Pixel> image3 = image2;
    dacpp::Tensor<Pixel, 2> image_tensor3({height, width}, image3);


    // 执行亮度增强操作
    imageAdjustment(image_tensor2, image_tensor3) <-> image_2;
    std::cout << "\nImage After Brightness Enhancement:" << std::endl;
    image_tensor3.print();

    return 0;
}
