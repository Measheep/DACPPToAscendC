//  自动生成的 AscendC Vector Host 代码
#include "dacpp_ascendc_host_runtime.h"
static_assert(DACPP_ASCENDC_HOST_RUNTIME_VERSION == 19,
              "regenerate Host runtime");

#include <type_traits>

#ifndef ASCENDC_CPU_DEBUG
extern "C" uint32_t vector_kernel_0_do(uint32_t blockDim, void *stream, uint8_t *imageR, uint8_t *imageG, uint8_t *imageB, uint8_t *image2R, uint8_t *image2G, uint8_t *image2B, uint32_t totalLength, uint32_t coreNum, uint32_t tileLength, uint32_t tileCount);
#else
extern "C" __global__ __aicore__ void vector_kernel_0(GM_ADDR imageR, GM_ADDR imageG, GM_ADDR imageB, GM_ADDR image2R, GM_ADDR image2G, GM_ADDR image2B, uint32_t totalLength, uint32_t coreNum, uint32_t tileLength, uint32_t tileCount);
#endif
#ifndef ASCENDC_CPU_DEBUG
extern "C" uint32_t vector_kernel_1_do(uint32_t blockDim, void *stream, uint8_t *imageR, uint8_t *imageG, uint8_t *imageB, uint8_t *image2R, uint8_t *image2G, uint8_t *image2B, uint32_t totalLength, uint32_t coreNum, uint32_t tileLength, uint32_t tileCount);
#else
extern "C" __global__ __aicore__ void vector_kernel_1(GM_ADDR imageR, GM_ADDR imageG, GM_ADDR imageB, GM_ADDR image2R, GM_ADDR image2G, GM_ADDR image2B, uint32_t totalLength, uint32_t coreNum, uint32_t tileLength, uint32_t tileCount);
#endif

struct Pixel {
    int r, g, b;
    // 重载 << 操作符，使 Pixel 类型的对象可以被输出
    friend std::ostream& operator<<(std::ostream& os, const Pixel& pixel) {
        os << "(" << pixel.r << ", " << pixel.g << ", " << pixel.b << ")";
        return os;
    }
};

int32_t main(int32_t argc, char *argv[]) {
    dacpp_host::Session app;
#ifndef ASCENDC_CPU_DEBUG
    dacpp_host::Device dev(app, 0);
    if (!dev.ok()) return app.finish();
    aclrtStream stream = dev.stream();
#endif
    const int width = 1920;
    const int height = 1080 * 8;
    std::vector<struct Pixel> image(height * width, {100, 90, 80});
    std::vector<struct Pixel> image2(height * width, {1, 2, 3});
    std::cout << "Original Image:" << std::endl;
    {
    //  算子 vector_kernel_0 基础参数与数据定义 
        uint32_t blockDim0 = 8;
        dacpp_host::HostBuffer<int32_t> imageRBuf(app, image.size(), static_cast<size_t>(0), static_cast<size_t>(0), static_cast<size_t>((256 + sizeof(int32_t) - 1) / sizeof(int32_t)));
        for (size_t i = 0; i < std::min(imageRBuf.logicalElems, image.size()); ++i) {
            imageRBuf.data[i] = static_cast<int32_t>(image[i].r);
        }
        dacpp_host::HostBuffer<int32_t> imageGBuf(app, image.size(), static_cast<size_t>(0), static_cast<size_t>(0), static_cast<size_t>((256 + sizeof(int32_t) - 1) / sizeof(int32_t)));
        for (size_t i = 0; i < std::min(imageGBuf.logicalElems, image.size()); ++i) {
            imageGBuf.data[i] = static_cast<int32_t>(image[i].g);
        }
        dacpp_host::HostBuffer<int32_t> imageBBuf(app, image.size(), static_cast<size_t>(0), static_cast<size_t>(0), static_cast<size_t>((256 + sizeof(int32_t) - 1) / sizeof(int32_t)));
        for (size_t i = 0; i < std::min(imageBBuf.logicalElems, image.size()); ++i) {
            imageBBuf.data[i] = static_cast<int32_t>(image[i].b);
        }
        dacpp_host::ensureSize(image2, static_cast<size_t>(16588800));
        dacpp_host::HostBuffer<int32_t> image2RBuf(app, image2.size(), static_cast<size_t>(0), static_cast<size_t>(0), static_cast<size_t>((256 + sizeof(int32_t) - 1) / sizeof(int32_t)));
        dacpp_host::HostBuffer<int32_t> image2GBuf(app, image2.size(), static_cast<size_t>(0), static_cast<size_t>(0), static_cast<size_t>((256 + sizeof(int32_t) - 1) / sizeof(int32_t)));
        dacpp_host::HostBuffer<int32_t> image2BBuf(app, image2.size(), static_cast<size_t>(0), static_cast<size_t>(0), static_cast<size_t>((256 + sizeof(int32_t) - 1) / sizeof(int32_t)));
        size_t kernelSize0 = static_cast<size_t>(0);
        const size_t coreCount0 = blockDim0 == 0 ? static_cast<size_t>(1) : static_cast<size_t>(blockDim0);
        kernelSize0 = std::max(kernelSize0, imageRBuf.kernelElems);
        kernelSize0 = std::max(kernelSize0, imageGBuf.kernelElems);
        kernelSize0 = std::max(kernelSize0, imageBBuf.kernelElems);
        kernelSize0 = std::max(kernelSize0, image2RBuf.kernelElems);
        kernelSize0 = std::max(kernelSize0, image2GBuf.kernelElems);
        kernelSize0 = std::max(kernelSize0, image2BBuf.kernelElems);
        uint32_t kernelLength0 = 0;
        const dacpp_host::Tile vectorRuntimeTiling0 = dacpp_host::makeTile(app, kernelSize0, coreCount0, 64, 2368);
        kernelSize0 = static_cast<size_t>(vectorRuntimeTiling0.len);
        kernelLength0 = vectorRuntimeTiling0.len;
        uint32_t tileLength0 = vectorRuntimeTiling0.size;
        uint32_t tileCount0 = vectorRuntimeTiling0.count;
        imageRBuf.kernelElems = std::max(imageRBuf.kernelElems, kernelSize0);
        { size_t withLeftHalo = 0, storage = 0, physical = 0; dacpp_host::checkedAdd(app, imageRBuf.kernelElems, imageRBuf.leftHaloElems, withLeftHalo); dacpp_host::checkedAdd(app, withLeftHalo, imageRBuf.rightHaloElems, storage); dacpp_host::checkedAlign(app, storage, imageRBuf.alignElems, physical); if (app.ok() && imageRBuf.data.size() < physical) imageRBuf.data.resize(physical); }
        imageGBuf.kernelElems = std::max(imageGBuf.kernelElems, kernelSize0);
        { size_t withLeftHalo = 0, storage = 0, physical = 0; dacpp_host::checkedAdd(app, imageGBuf.kernelElems, imageGBuf.leftHaloElems, withLeftHalo); dacpp_host::checkedAdd(app, withLeftHalo, imageGBuf.rightHaloElems, storage); dacpp_host::checkedAlign(app, storage, imageGBuf.alignElems, physical); if (app.ok() && imageGBuf.data.size() < physical) imageGBuf.data.resize(physical); }
        imageBBuf.kernelElems = std::max(imageBBuf.kernelElems, kernelSize0);
        { size_t withLeftHalo = 0, storage = 0, physical = 0; dacpp_host::checkedAdd(app, imageBBuf.kernelElems, imageBBuf.leftHaloElems, withLeftHalo); dacpp_host::checkedAdd(app, withLeftHalo, imageBBuf.rightHaloElems, storage); dacpp_host::checkedAlign(app, storage, imageBBuf.alignElems, physical); if (app.ok() && imageBBuf.data.size() < physical) imageBBuf.data.resize(physical); }
        image2RBuf.kernelElems = std::max(image2RBuf.kernelElems, kernelSize0);
        { size_t withLeftHalo = 0, storage = 0, physical = 0; dacpp_host::checkedAdd(app, image2RBuf.kernelElems, image2RBuf.leftHaloElems, withLeftHalo); dacpp_host::checkedAdd(app, withLeftHalo, image2RBuf.rightHaloElems, storage); dacpp_host::checkedAlign(app, storage, image2RBuf.alignElems, physical); if (app.ok() && image2RBuf.data.size() < physical) image2RBuf.data.resize(physical); }
        image2GBuf.kernelElems = std::max(image2GBuf.kernelElems, kernelSize0);
        { size_t withLeftHalo = 0, storage = 0, physical = 0; dacpp_host::checkedAdd(app, image2GBuf.kernelElems, image2GBuf.leftHaloElems, withLeftHalo); dacpp_host::checkedAdd(app, withLeftHalo, image2GBuf.rightHaloElems, storage); dacpp_host::checkedAlign(app, storage, image2GBuf.alignElems, physical); if (app.ok() && image2GBuf.data.size() < physical) image2GBuf.data.resize(physical); }
        image2BBuf.kernelElems = std::max(image2BBuf.kernelElems, kernelSize0);
        { size_t withLeftHalo = 0, storage = 0, physical = 0; dacpp_host::checkedAdd(app, image2BBuf.kernelElems, image2BBuf.leftHaloElems, withLeftHalo); dacpp_host::checkedAdd(app, withLeftHalo, image2BBuf.rightHaloElems, storage); dacpp_host::checkedAlign(app, storage, image2BBuf.alignElems, physical); if (app.ok() && image2BBuf.data.size() < physical) image2BBuf.data.resize(physical); }
    //  算子 vector_kernel_0 NPU 执行流 
    #ifndef ASCENDC_CPU_DEBUG
        dev.allocate(imageRBuf);
        dev.allocate(imageGBuf);
        dev.allocate(imageBBuf);
        dev.allocate(image2RBuf);
        dev.allocate(image2GBuf);
        dev.allocate(image2BBuf);
    #else
        uint8_t *cpuGm_imageR = (uint8_t *)AscendC::GmAlloc(imageRBuf.bytes());
        uint8_t *cpuGm_imageG = (uint8_t *)AscendC::GmAlloc(imageGBuf.bytes());
        uint8_t *cpuGm_imageB = (uint8_t *)AscendC::GmAlloc(imageBBuf.bytes());
        uint8_t *cpuGm_image2R = (uint8_t *)AscendC::GmAlloc(image2RBuf.bytes());
        uint8_t *cpuGm_image2G = (uint8_t *)AscendC::GmAlloc(image2GBuf.bytes());
        uint8_t *cpuGm_image2B = (uint8_t *)AscendC::GmAlloc(image2BBuf.bytes());
    #endif
    #ifndef ASCENDC_CPU_DEBUG
        dev.copyIn(imageRBuf);
        dev.copyIn(imageGBuf);
        dev.copyIn(imageBBuf);
    #else
        memcpy(cpuGm_imageR, imageRBuf.data.data(), imageRBuf.bytes());
        memcpy(cpuGm_imageG, imageGBuf.data.data(), imageGBuf.bytes());
        memcpy(cpuGm_imageB, imageBBuf.data.data(), imageBBuf.bytes());
    #endif
    #ifndef ASCENDC_CPU_DEBUG
        dev.launch(vector_kernel_0_do(blockDim0, stream, imageRBuf.device, imageGBuf.device, imageBBuf.device, image2RBuf.device, image2GBuf.device, image2BBuf.device, kernelLength0, blockDim0, tileLength0, tileCount0));
        dev.sync();
    #else
        ICPU_RUN_KF(vector_kernel_0, blockDim0, cpuGm_imageR, cpuGm_imageG, cpuGm_imageB, cpuGm_image2R, cpuGm_image2G, cpuGm_image2B, kernelLength0, blockDim0, tileLength0, tileCount0);
    #endif
    #ifndef ASCENDC_CPU_DEBUG
        dev.copyOut(image2RBuf);
        for (size_t i = 0; i < image2RBuf.logicalElems; ++i) {
            auto &elem = image2[i];
            using image2RFieldType = typename std::remove_reference<decltype(elem.r)>::type;
            elem.r = static_cast<image2RFieldType>(image2RBuf.data[i]);
        }
        dev.copyOut(image2GBuf);
        for (size_t i = 0; i < image2GBuf.logicalElems; ++i) {
            auto &elem = image2[i];
            using image2GFieldType = typename std::remove_reference<decltype(elem.g)>::type;
            elem.g = static_cast<image2GFieldType>(image2GBuf.data[i]);
        }
        dev.copyOut(image2BBuf);
        for (size_t i = 0; i < image2BBuf.logicalElems; ++i) {
            auto &elem = image2[i];
            using image2BFieldType = typename std::remove_reference<decltype(elem.b)>::type;
            elem.b = static_cast<image2BFieldType>(image2BBuf.data[i]);
        }
    #else
        memcpy(image2RBuf.data.data(), cpuGm_image2R, image2RBuf.bytes());
        for (size_t i = 0; i < image2RBuf.logicalElems; ++i) {
            auto &elem = image2[i];
            using image2RFieldType = typename std::remove_reference<decltype(elem.r)>::type;
            elem.r = static_cast<image2RFieldType>(image2RBuf.data[i]);
        }
        memcpy(image2GBuf.data.data(), cpuGm_image2G, image2GBuf.bytes());
        for (size_t i = 0; i < image2GBuf.logicalElems; ++i) {
            auto &elem = image2[i];
            using image2GFieldType = typename std::remove_reference<decltype(elem.g)>::type;
            elem.g = static_cast<image2GFieldType>(image2GBuf.data[i]);
        }
        memcpy(image2BBuf.data.data(), cpuGm_image2B, image2BBuf.bytes());
        for (size_t i = 0; i < image2BBuf.logicalElems; ++i) {
            auto &elem = image2[i];
            using image2BFieldType = typename std::remove_reference<decltype(elem.b)>::type;
            elem.b = static_cast<image2BFieldType>(image2BBuf.data[i]);
        }
    #endif
    #ifndef ASCENDC_CPU_DEBUG
        dev.release(imageRBuf);
        dev.release(imageGBuf);
        dev.release(imageBBuf);
        dev.release(image2RBuf);
        dev.release(image2GBuf);
        dev.release(image2BBuf);
    #else
        AscendC::GmFree((void *)cpuGm_imageR);
        AscendC::GmFree((void *)cpuGm_imageG);
        AscendC::GmFree((void *)cpuGm_imageB);
        AscendC::GmFree((void *)cpuGm_image2R);
        AscendC::GmFree((void *)cpuGm_image2G);
        AscendC::GmFree((void *)cpuGm_image2B);
    #endif
    }
    std::cout << "\nImage After Color Adjustment:" << std::endl;
    std::vector<struct Pixel> image3(image2);
    {
    //  算子 vector_kernel_1 基础参数与数据定义 
        uint32_t blockDim1 = 8;
        dacpp_host::HostBuffer<int32_t> imageRBuf(app, image2.size(), static_cast<size_t>(0), static_cast<size_t>(0), static_cast<size_t>((256 + sizeof(int32_t) - 1) / sizeof(int32_t)));
        for (size_t i = 0; i < std::min(imageRBuf.logicalElems, image2.size()); ++i) {
            imageRBuf.data[i] = static_cast<int32_t>(image2[i].r);
        }
        dacpp_host::HostBuffer<int32_t> imageGBuf(app, image2.size(), static_cast<size_t>(0), static_cast<size_t>(0), static_cast<size_t>((256 + sizeof(int32_t) - 1) / sizeof(int32_t)));
        for (size_t i = 0; i < std::min(imageGBuf.logicalElems, image2.size()); ++i) {
            imageGBuf.data[i] = static_cast<int32_t>(image2[i].g);
        }
        dacpp_host::HostBuffer<int32_t> imageBBuf(app, image2.size(), static_cast<size_t>(0), static_cast<size_t>(0), static_cast<size_t>((256 + sizeof(int32_t) - 1) / sizeof(int32_t)));
        for (size_t i = 0; i < std::min(imageBBuf.logicalElems, image2.size()); ++i) {
            imageBBuf.data[i] = static_cast<int32_t>(image2[i].b);
        }
        dacpp_host::ensureSize(image3, static_cast<size_t>(16588800));
        dacpp_host::HostBuffer<int32_t> image2RBuf(app, image3.size(), static_cast<size_t>(0), static_cast<size_t>(0), static_cast<size_t>((256 + sizeof(int32_t) - 1) / sizeof(int32_t)));
        dacpp_host::HostBuffer<int32_t> image2GBuf(app, image3.size(), static_cast<size_t>(0), static_cast<size_t>(0), static_cast<size_t>((256 + sizeof(int32_t) - 1) / sizeof(int32_t)));
        dacpp_host::HostBuffer<int32_t> image2BBuf(app, image3.size(), static_cast<size_t>(0), static_cast<size_t>(0), static_cast<size_t>((256 + sizeof(int32_t) - 1) / sizeof(int32_t)));
        size_t kernelSize1 = static_cast<size_t>(0);
        const size_t coreCount1 = blockDim1 == 0 ? static_cast<size_t>(1) : static_cast<size_t>(blockDim1);
        kernelSize1 = std::max(kernelSize1, imageRBuf.kernelElems);
        kernelSize1 = std::max(kernelSize1, imageGBuf.kernelElems);
        kernelSize1 = std::max(kernelSize1, imageBBuf.kernelElems);
        kernelSize1 = std::max(kernelSize1, image2RBuf.kernelElems);
        kernelSize1 = std::max(kernelSize1, image2GBuf.kernelElems);
        kernelSize1 = std::max(kernelSize1, image2BBuf.kernelElems);
        uint32_t kernelLength1 = 0;
        const dacpp_host::Tile vectorRuntimeTiling1 = dacpp_host::makeTile(app, kernelSize1, coreCount1, 64, 2368);
        kernelSize1 = static_cast<size_t>(vectorRuntimeTiling1.len);
        kernelLength1 = vectorRuntimeTiling1.len;
        uint32_t tileLength1 = vectorRuntimeTiling1.size;
        uint32_t tileCount1 = vectorRuntimeTiling1.count;
        imageRBuf.kernelElems = std::max(imageRBuf.kernelElems, kernelSize1);
        { size_t withLeftHalo = 0, storage = 0, physical = 0; dacpp_host::checkedAdd(app, imageRBuf.kernelElems, imageRBuf.leftHaloElems, withLeftHalo); dacpp_host::checkedAdd(app, withLeftHalo, imageRBuf.rightHaloElems, storage); dacpp_host::checkedAlign(app, storage, imageRBuf.alignElems, physical); if (app.ok() && imageRBuf.data.size() < physical) imageRBuf.data.resize(physical); }
        imageGBuf.kernelElems = std::max(imageGBuf.kernelElems, kernelSize1);
        { size_t withLeftHalo = 0, storage = 0, physical = 0; dacpp_host::checkedAdd(app, imageGBuf.kernelElems, imageGBuf.leftHaloElems, withLeftHalo); dacpp_host::checkedAdd(app, withLeftHalo, imageGBuf.rightHaloElems, storage); dacpp_host::checkedAlign(app, storage, imageGBuf.alignElems, physical); if (app.ok() && imageGBuf.data.size() < physical) imageGBuf.data.resize(physical); }
        imageBBuf.kernelElems = std::max(imageBBuf.kernelElems, kernelSize1);
        { size_t withLeftHalo = 0, storage = 0, physical = 0; dacpp_host::checkedAdd(app, imageBBuf.kernelElems, imageBBuf.leftHaloElems, withLeftHalo); dacpp_host::checkedAdd(app, withLeftHalo, imageBBuf.rightHaloElems, storage); dacpp_host::checkedAlign(app, storage, imageBBuf.alignElems, physical); if (app.ok() && imageBBuf.data.size() < physical) imageBBuf.data.resize(physical); }
        image2RBuf.kernelElems = std::max(image2RBuf.kernelElems, kernelSize1);
        { size_t withLeftHalo = 0, storage = 0, physical = 0; dacpp_host::checkedAdd(app, image2RBuf.kernelElems, image2RBuf.leftHaloElems, withLeftHalo); dacpp_host::checkedAdd(app, withLeftHalo, image2RBuf.rightHaloElems, storage); dacpp_host::checkedAlign(app, storage, image2RBuf.alignElems, physical); if (app.ok() && image2RBuf.data.size() < physical) image2RBuf.data.resize(physical); }
        image2GBuf.kernelElems = std::max(image2GBuf.kernelElems, kernelSize1);
        { size_t withLeftHalo = 0, storage = 0, physical = 0; dacpp_host::checkedAdd(app, image2GBuf.kernelElems, image2GBuf.leftHaloElems, withLeftHalo); dacpp_host::checkedAdd(app, withLeftHalo, image2GBuf.rightHaloElems, storage); dacpp_host::checkedAlign(app, storage, image2GBuf.alignElems, physical); if (app.ok() && image2GBuf.data.size() < physical) image2GBuf.data.resize(physical); }
        image2BBuf.kernelElems = std::max(image2BBuf.kernelElems, kernelSize1);
        { size_t withLeftHalo = 0, storage = 0, physical = 0; dacpp_host::checkedAdd(app, image2BBuf.kernelElems, image2BBuf.leftHaloElems, withLeftHalo); dacpp_host::checkedAdd(app, withLeftHalo, image2BBuf.rightHaloElems, storage); dacpp_host::checkedAlign(app, storage, image2BBuf.alignElems, physical); if (app.ok() && image2BBuf.data.size() < physical) image2BBuf.data.resize(physical); }
    //  算子 vector_kernel_1 NPU 执行流 
    #ifndef ASCENDC_CPU_DEBUG
        dev.allocate(imageRBuf);
        dev.allocate(imageGBuf);
        dev.allocate(imageBBuf);
        dev.allocate(image2RBuf);
        dev.allocate(image2GBuf);
        dev.allocate(image2BBuf);
    #else
        uint8_t *cpuGm_imageR = (uint8_t *)AscendC::GmAlloc(imageRBuf.bytes());
        uint8_t *cpuGm_imageG = (uint8_t *)AscendC::GmAlloc(imageGBuf.bytes());
        uint8_t *cpuGm_imageB = (uint8_t *)AscendC::GmAlloc(imageBBuf.bytes());
        uint8_t *cpuGm_image2R = (uint8_t *)AscendC::GmAlloc(image2RBuf.bytes());
        uint8_t *cpuGm_image2G = (uint8_t *)AscendC::GmAlloc(image2GBuf.bytes());
        uint8_t *cpuGm_image2B = (uint8_t *)AscendC::GmAlloc(image2BBuf.bytes());
    #endif
    #ifndef ASCENDC_CPU_DEBUG
        dev.copyIn(imageRBuf);
        dev.copyIn(imageGBuf);
        dev.copyIn(imageBBuf);
    #else
        memcpy(cpuGm_imageR, imageRBuf.data.data(), imageRBuf.bytes());
        memcpy(cpuGm_imageG, imageGBuf.data.data(), imageGBuf.bytes());
        memcpy(cpuGm_imageB, imageBBuf.data.data(), imageBBuf.bytes());
    #endif
    #ifndef ASCENDC_CPU_DEBUG
        dev.launch(vector_kernel_1_do(blockDim1, stream, imageRBuf.device, imageGBuf.device, imageBBuf.device, image2RBuf.device, image2GBuf.device, image2BBuf.device, kernelLength1, blockDim1, tileLength1, tileCount1));
        dev.sync();
    #else
        ICPU_RUN_KF(vector_kernel_1, blockDim1, cpuGm_imageR, cpuGm_imageG, cpuGm_imageB, cpuGm_image2R, cpuGm_image2G, cpuGm_image2B, kernelLength1, blockDim1, tileLength1, tileCount1);
    #endif
    #ifndef ASCENDC_CPU_DEBUG
        dev.copyOut(image2RBuf);
        for (size_t i = 0; i < image2RBuf.logicalElems; ++i) {
            auto &elem = image3[i];
            using image2RFieldType = typename std::remove_reference<decltype(elem.r)>::type;
            elem.r = static_cast<image2RFieldType>(image2RBuf.data[i]);
        }
        dev.copyOut(image2GBuf);
        for (size_t i = 0; i < image2GBuf.logicalElems; ++i) {
            auto &elem = image3[i];
            using image2GFieldType = typename std::remove_reference<decltype(elem.g)>::type;
            elem.g = static_cast<image2GFieldType>(image2GBuf.data[i]);
        }
        dev.copyOut(image2BBuf);
        for (size_t i = 0; i < image2BBuf.logicalElems; ++i) {
            auto &elem = image3[i];
            using image2BFieldType = typename std::remove_reference<decltype(elem.b)>::type;
            elem.b = static_cast<image2BFieldType>(image2BBuf.data[i]);
        }
    #else
        memcpy(image2RBuf.data.data(), cpuGm_image2R, image2RBuf.bytes());
        for (size_t i = 0; i < image2RBuf.logicalElems; ++i) {
            auto &elem = image3[i];
            using image2RFieldType = typename std::remove_reference<decltype(elem.r)>::type;
            elem.r = static_cast<image2RFieldType>(image2RBuf.data[i]);
        }
        memcpy(image2GBuf.data.data(), cpuGm_image2G, image2GBuf.bytes());
        for (size_t i = 0; i < image2GBuf.logicalElems; ++i) {
            auto &elem = image3[i];
            using image2GFieldType = typename std::remove_reference<decltype(elem.g)>::type;
            elem.g = static_cast<image2GFieldType>(image2GBuf.data[i]);
        }
        memcpy(image2BBuf.data.data(), cpuGm_image2B, image2BBuf.bytes());
        for (size_t i = 0; i < image2BBuf.logicalElems; ++i) {
            auto &elem = image3[i];
            using image2BFieldType = typename std::remove_reference<decltype(elem.b)>::type;
            elem.b = static_cast<image2BFieldType>(image2BBuf.data[i]);
        }
    #endif
    #ifndef ASCENDC_CPU_DEBUG
        dev.release(imageRBuf);
        dev.release(imageGBuf);
        dev.release(imageBBuf);
        dev.release(image2RBuf);
        dev.release(image2GBuf);
        dev.release(image2BBuf);
    #else
        AscendC::GmFree((void *)cpuGm_imageR);
        AscendC::GmFree((void *)cpuGm_imageG);
        AscendC::GmFree((void *)cpuGm_imageB);
        AscendC::GmFree((void *)cpuGm_image2R);
        AscendC::GmFree((void *)cpuGm_image2G);
        AscendC::GmFree((void *)cpuGm_image2B);
    #endif
    }
    std::cout << "\nImage After Brightness Enhancement:" << std::endl;
    {
        std::cout << "{";
        for (long long dacppPrintI0_0 = 0; dacppPrintI0_0 < static_cast<long long>(height); ++dacppPrintI0_0) {
            if (dacppPrintI0_0 != 0) { std::cout << ", "; }
            std::cout << "{";
            for (long long dacppPrintI0_1 = 0; dacppPrintI0_1 < static_cast<long long>(width); ++dacppPrintI0_1) {
                if (dacppPrintI0_1 != 0) { std::cout << ", "; }
                std::cout << (image3.data())[static_cast<long long>(0) + dacppPrintI0_0 * static_cast<long long>(width) + dacppPrintI0_1 * static_cast<long long>(1)];
            }
            std::cout << "}";
        }
        std::cout << "}";
        std::cout << std::endl;
    }
#ifndef ASCENDC_CPU_DEBUG
    dev.close();
#endif
    return app.finish();
}

