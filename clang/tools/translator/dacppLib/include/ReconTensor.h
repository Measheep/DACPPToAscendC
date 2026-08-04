//lsj优化后的Tensor 优化数据传输与初始化 通过大规模buffer模型测试
#ifndef RECONTENSOR_H
#define RECONTENSOR_H

#include <algorithm>
#include <iostream>
#include <vector>
#include <exception>
#include <memory>
#include "Slice.h"
#include "FuncTensor.hpp"
#include <cstring>
#include <vector>
#include <type_traits>
#include <iomanip> // 记得包含这个头文件
#include <chrono>     // for std::chrono


namespace dacpp {
    template <typename Func>
    inline void dac_for(int time_steps, Func&& loop_body) {
        loop_body(time_steps);
    }
    template <typename T1, typename U1>
    void swap(const T1&, const U1&) {
        // 什么都不做，避免真正执行
    }
     
    template <class T, int N>
    class TensorProxy;
    template <class T>
    class TensorProxy <T, 1>;
  
    template<class ImplType>
    class TensorBase{
    public:
        operator FuncTensor<ImplType>() const {
            return FuncTensor<ImplType>(this->data_, this->offset_, this->dim_, this->shape_, this->stride_);
        };
        std::shared_ptr<ImplType> getDataPtr() const;
         int getOffset() const;
        int getDim() const;
        std::shared_ptr<int> getShapePtr() const;
        std::shared_ptr<int> getStridePtr() const;
        int getShape(int dimIdx) const;
        int getStride(int dimIdx) const;
        int getSize() const;
        void tensor2Array(ImplType* data) const;
        void tensor2Array(std::vector <ImplType>& data)const;
        void array2Tensor(ImplType* data);
        void array2Tensor(std::vector <ImplType> data);
        void print() const;
        int getCurrentDim() const;
        ImplType getElement(std::vector<int> indices) const ;
        void reviseValue(ImplType val, std::vector<int> indices);
    protected:
        void NextDim();
        void recursiveTake(ImplType* data, int& idx, int dimIdx) const;
        void recursiveTake(std::vector <ImplType>& data, int& idx, int dimIdx) const;
        void recursiveBring(ImplType* data, int& idx, int dimIdx);
        void recursiveBring(std::vector <ImplType> data, int& idx, int dimIdx);
        void recursivePrint(int dimIdx) const;
        std::shared_ptr<ImplType> data_;
        int offset_;
        int dim_;
        std::shared_ptr<int> shape_;
        std::shared_ptr<int> stride_;
        int current_dim = 0;
        std::vector<ImplType> tmp_data;
    };
    template <class ImplType>
    int TensorBase<ImplType> :: getStride(int dimIdx) const {return this->stride_.get()[dimIdx];}
    template <class ImplType>
    int TensorBase<ImplType> :: getShape(int dimIdx) const {return this->shape_.get()[dimIdx]; }
    template <class ImplType>
    std::shared_ptr<int> TensorBase<ImplType> :: getStridePtr() const {return this->stride_;}
    template <class ImplType>
    std::shared_ptr<ImplType> TensorBase<ImplType> :: getDataPtr() const{return this->data_;}
    template <class ImplType>
    std::shared_ptr<int> TensorBase<ImplType> :: getShapePtr() const {return this->shape_;}
    template <class ImplType>
    int TensorBase<ImplType> :: getOffset() const {return this->offset_;}
    template <class ImplType>
    int TensorBase<ImplType> :: getDim() const {return this->dim_;}
 
    template<class ImplType>
    int TensorBase<ImplType> :: getSize() const {
        int size = 1;
        for(int dimIdx = 0; dimIdx < getDim(); dimIdx++) {
            size *= this->shape_.get()[dimIdx];
        }
        return size;
    }
    template<class ImplType>
    void TensorBase<ImplType> :: tensor2Array(ImplType* data) const { 
    const int total_size = getSize();
    const ImplType* src = this->data_.get();
    const int offset = this->offset_;
    const int* stride = this->stride_.get();
    const int* shape  = this->shape_.get();
    const int dim = this->dim_;

    // 判断是否连续存储
    bool is_contiguous = true;
    int expected_stride = 1;
    for (int i = dim - 1; i >= 0; --i) {
        if (stride[i] != expected_stride) {
            is_contiguous = false;
            break;
        }
        expected_stride *= shape[i];
    }

    // 连续存储：直接 memcpy
    if (is_contiguous) {
        std::memcpy(data, src + offset, sizeof(ImplType) * total_size);
        return;
    }

    // 非连续存储：线性访问转换
    std::vector<int> idx(dim, 0);
    for (int linear = 0; linear < total_size; ++linear) {
        int real_index = offset;
        for (int d = 0; d < dim; ++d) {
            real_index += idx[d] * stride[d];
        }
        data[linear] = src[real_index];

        // 增加多维坐标
        for (int d = dim - 1; d >= 0; --d) {
            idx[d]++;
            if (idx[d] < shape[d]) break;
            idx[d] = 0;
        }
    }
}
    template<class ImplType>
    void TensorBase<ImplType>::tensor2Array(std::vector<ImplType>& data) const {
    const int total_size = getSize();
    data.resize(total_size); // 调整输出 vector 大小

    ImplType* dst = data.data();                // 输出指针
    const ImplType* src = this->data_.get();    // 源指针
    const int offset = this->offset_;
    const int* stride = this->stride_.get();
    const int* shape  = this->shape_.get();
    const int dim = this->dim_;

    // 判断是否连续存储
    bool is_contiguous = true;
    int expected_stride = 1;
    for (int i = dim - 1; i >= 0; --i) {
        if (stride[i] != expected_stride) {
            is_contiguous = false;
            break;
        }
        expected_stride *= shape[i];
    }

    // 连续存储：直接 memcpy
    if (is_contiguous) {
        std::memcpy(dst, src + offset, sizeof(ImplType) * total_size);
    } else {
        // 非连续存储：线性访问转换
        std::vector<int> idx(dim, 0);
        for (int linear = 0; linear < total_size; ++linear) {
            int real_index = offset;
            for (int d = 0; d < dim; ++d) {
                real_index += idx[d] * stride[d];
            }
            dst[linear] = src[real_index];

            // 多维索引递增
            for (int d = dim - 1; d >= 0; --d) {
                idx[d]++;
                if (idx[d] < shape[d]) break;
                idx[d] = 0;
            }
        }
    }
}
    template<class ImplType>
    void TensorBase<ImplType> :: array2Tensor(ImplType* data) {
    const int total_size = getSize();
    ImplType* dst = this->data_.get();
    const int offset = this->offset_;
    const int* stride = this->stride_.get();
    const int* shape  = this->shape_.get();
    const int dim = this->dim_;

    // 判断是否连续存储
    bool is_contiguous = true;
    int expected_stride = 1;
    for (int i = dim - 1; i >= 0; --i) {
        if (stride[i] != expected_stride) {
            is_contiguous = false;
            break;
        }
        expected_stride *= shape[i];
    }

    // 连续存储：直接 memcpy
    if (is_contiguous) {
        std::memcpy(dst + offset, data, sizeof(ImplType) * total_size);
        return;
    }
 
    // 非连续存储：线性访问转换
    std::vector<int> idx(dim, 0);
    for (int linear = 0; linear < total_size; ++linear) {
        int real_index = offset;
        for (int d = 0; d < dim; ++d) {
            real_index += idx[d] * stride[d];
        }
        dst[real_index] = data[linear];

        // 增加多维坐标
        for (int d = dim - 1; d >= 0; --d) {
            idx[d]++;
            if (idx[d] < shape[d]) break;
            idx[d] = 0;
        }
    }
}
    template<class ImplType>
    void TensorBase<ImplType>::array2Tensor(std::vector<ImplType> data) {

    const int total_size = getSize();
    ImplType* dst = this->data_.get();   // Tensor 内存指针
    const int offset = this->offset_;
    const int* stride = this->stride_.get();
    const int* shape  = this->shape_.get();
    const int dim = this->dim_;

    // 判断是否连续存储
    bool is_contiguous = true;
    int expected_stride = 1;
    for (int i = dim - 1; i >= 0; --i) {
        if (stride[i] != expected_stride) {
            is_contiguous = false;
            break;
        }
        expected_stride *= shape[i];
    }

    // 连续存储：直接 memcpy
    if (is_contiguous) {
        std::memcpy(dst + offset, data.data(), sizeof(ImplType) * total_size);
    } else {
        // 非连续存储：线性访问转换
        std::vector<int> idx(dim, 0);
        for (int linear = 0; linear < total_size; ++linear) {
            int real_index = offset;
            for (int d = 0; d < dim; ++d) {
                real_index += idx[d] * stride[d];
            }
            dst[real_index] = data[linear];

            // 多维索引递增
            for (int d = dim - 1; d >= 0; --d) {
                idx[d]++;
                if (idx[d] < shape[d]) break;
                idx[d] = 0;
            }
        }
    }

}
    template<class ImplType>
    void TensorBase<ImplType> :: print() const {
        recursivePrint(0);
        std::cout << "\n";
    }
    template<class ImplType>
    int TensorBase<ImplType> :: getCurrentDim() const {
        return this->current_dim;
    }
    template<class ImplType>
    void TensorBase<ImplType> :: NextDim(){
        this->current_dim ++;
    }
     template<class ImplType>
    void TensorBase<ImplType> :: recursiveTake(ImplType* data, int &idx, int dimIdx) const {
        static std::vector<int> indices;
        if(dimIdx == this->dim_) {
            int index = this->offset_;
            for(int i = 0; i < this->dim_; i++) 
                index += indices[i] * this->stride_.get()[i];
            data[idx++] = this->data_.get()[index];
            return;
        }
        for(int i = 0; i < this->shape_.get()[dimIdx]; i++) {
            indices.push_back(i);
            recursiveTake(data, idx, dimIdx + 1);
            indices.pop_back();
        }
    } 
    template<class ImplType>
    void TensorBase<ImplType> :: recursiveTake(std::vector <ImplType>& data, int &idx, int dimIdx) const {
        static std::vector<int> indices;
        if(dimIdx == this->dim_) {
            int index = this->offset_;
            for(int i = 0; i < this->dim_; i++) 
                index += indices[i] * this->stride_.get()[i];
            data[idx++]=this->data_.get()[index];
            return;
        }
        for(int i = 0; i < this->shape_.get()[dimIdx]; i++) {
            indices.push_back(i);
            recursiveTake(data, idx, dimIdx + 1);
            indices.pop_back();
        }
    }
    template<class ImplType>
    void TensorBase<ImplType> :: recursiveBring(ImplType* data, int &idx, int dimIdx) {
        static std::vector<int> indices;
        if(dimIdx == this->dim_) {
            int index = this->offset_;
            for(int i = 0; i < this->dim_; i++) 
                index += indices[i] * this->stride_.get()[i];
            this->data_.get()[index] = data[idx++];
            return;
        }
        for(int i = 0; i < this->shape_.get()[dimIdx]; i++) {
            indices.push_back(i);
            recursiveBring(data, idx, dimIdx + 1);
            indices.pop_back();
        }
    }
    template<class ImplType>
    void TensorBase<ImplType> :: recursiveBring(std::vector <ImplType> data, int &idx, int dimIdx) {
        static std::vector<int> indices;
        if(dimIdx == this->dim_) {
            int index = this->offset_;
            for(int i = 0; i < this->dim_; i++) 
                index += indices[i] * this->stride_.get()[i];
            //std::cout<<index<<" "<<this->data_.get()[index]<<" "<<data[idx]<<std::endl;
            this->data_.get()[index] = data[idx++];
            return;
        }
        for(int i = 0; i < this->shape_.get()[dimIdx]; i++) {
            indices.push_back(i);
            recursiveBring(data, idx, dimIdx + 1);
            indices.pop_back();
        }
    }
    template<class ImplType>
    void TensorBase<ImplType> :: recursivePrint(int dimIdx) const {
        static std::vector<int> indices;
        if(dimIdx == this->dim_) {
            int index = this->offset_;
            for(int i = 0; i < this->dim_; i++) 
                index += indices[i] * this->stride_.get()[i];
            std::cout << this->data_.get()[index];
            return;
        }
        std::cout << "{";  
        for(int i = 0; i < this->shape_.get()[dimIdx]; i++) {
            indices.push_back(i);
            recursivePrint(dimIdx + 1);
            if(i != this->shape_.get()[dimIdx] - 1) 
                std::cout << ", ";
            indices.pop_back();
        }
        std::cout << "}";
    }
    template<class ImplType>
    ImplType TensorBase<ImplType> ::getElement(std::vector<int> indices) const {
        int index = this->getOffset();
        for(int i=0;i<this->getDim();i++)
            index = index + this->getStride(i)*indices[i];
        ImplType val = this->getDataPtr().get()[index];
        return val;
    }
    template<class ImplType>
    void TensorBase<ImplType> ::reviseValue(ImplType val, std::vector<int> indices){
        int index = this->getOffset();
        for(int i=0;i<this->getDim();i++)
            index = index + this->getStride(i)*indices[i];
        this->getDataPtr().get()[index] = val;
        return;
    }

    template<class ImplType, int N>
    class Tensor: public TensorBase <ImplType>{
        friend class dacpp::TensorProxy<ImplType, N>;
        friend class dacpp::TensorProxy<ImplType, 1>;
    public:
    /***************************************************************** */
        void takeOwnership(int*& ptr) {
            this->data_.reset(ptr);
            ptr = nullptr;
        }

        void takeOwnership( std::vector<ImplType>& vec) {
            this->tmp_data = std::move(vec);
            this->data_.reset(this->tmp_data.data(), [](ImplType*){ /* 不删除数据 */ });
        }
    /************************************************************************************ */
        Tensor(){
            this->dim_ = N;
            this->shape_.reset(new int[N]);
            this->stride_.reset(new int[N]);
        };
        Tensor(const TensorProxy<ImplType, N> &x);
        Tensor(const TensorProxy<ImplType, N> &&x);
        Tensor(const Tensor<ImplType, N> &x);
        Tensor(const std::vector<int> values, ImplType*& data);
        Tensor(const std::vector<int> values, std::vector<ImplType>& data);
        Tensor(const std::vector<int> values, ImplType data = 0);
        Tensor(std::shared_ptr<ImplType> data, int offset, int dim, std::shared_ptr<int> shape, std::shared_ptr<int> stride);
        Tensor(std::shared_ptr<ImplType> data, int offset, int dim, std::shared_ptr<int> shape, std::shared_ptr<int> stride, int currentdim);
        Tensor<ImplType, N>& operator=(const Tensor<ImplType, N>& operand); 
        Tensor<ImplType, N>& operator=(const TensorProxy<ImplType, N>& operand){
            std::vector<ImplType> data;
            operand.tensor2Array(data);
            this->data_.reset(new ImplType[data.size()]);   
            this->dim_ = operand.getDim();
            this->offset_ = 0;
            this->current_dim = 0;
            this->shape_.reset(new int[this->dim_]);
            this->stride_.reset(new int[this->dim_]);
            for(int i = this->dim_ - 1; i >=0; i--){
                this->shape_.get()[i] = operand.getShape(i);
                if(i == this->dim_ - 1)
                    this->stride_.get()[i] = 1;
                else
                    this->stride_.get()[i] = this->stride_.get()[i + 1] * this->shape_.get()[i + 1];
            }
            for(int i = 0; i < data.size(); i++)
                this->data_.get()[i] = data[i];
            return *this;
        }
        Tensor<ImplType, N>& operator=(const TensorProxy<ImplType, N>&& operand){
            this->data_ = operand.getDataPtr();
            this->offset_ = operand.getOffset();
            this->dim_ = operand.getDim();
            this->stride_ = operand.getStridePtr();
            this->shape_ = operand.getShapePtr();
            this->current_dim = 0;
            return *this;
        }
        TensorProxy<ImplType, N> operator[](std::initializer_list<int> idx);
        TensorProxy<ImplType, N-1> operator[](int idx) const;
        TensorProxy<ImplType, N> operator[](split sp) const;
        TensorProxy<ImplType, N - 1> operator[](index sp) const;
        Tensor<ImplType, N-1> slice(int dimIdx, int idx) const;
        Tensor<ImplType, N> slice(int dimIdx, int start, int end, int sliceStride = 1 , int ModifyDim = 0) const;
        Tensor<ImplType, N> operator+(const Tensor<ImplType, N>& operand) const{};
        Tensor<ImplType, N> operator-(const Tensor<ImplType, N>& operand) const{};
        Tensor<ImplType, N> operator*(const Tensor<ImplType, N>& operand) const{};
        Tensor<ImplType, N> operator/(const Tensor<ImplType, N>& operand) const{};
        Tensor<ImplType, N> operator%(const Tensor<ImplType, N>& operand) const{};
        void operator+=(const Tensor<ImplType, N>& operand){};
        void operator-=(const Tensor<ImplType, N>& operand){};
        void operator*=(const Tensor<ImplType, N>& operand){};
        void operator/=(const Tensor<ImplType, N>& operand){};
        void operator%=(const Tensor<ImplType, N>& operand){};
    };
    #define Base Tensor<ImplType, N>

    template<class ImplType, int N>
    Tensor<ImplType, N> :: Tensor(const TensorProxy<ImplType, N> &&x){
        this->data_ = x.getDataPtr();
        this->offset_ = x.getOffset();
        this->dim_ = x.getDim();
        this->stride_ = x.getStridePtr();
        this->shape_ = x.getShapePtr();
        this->current_dim = 0;
    }
    template<class ImplType, int N>
    Tensor<ImplType, N> :: Tensor(const TensorProxy<ImplType, N> &x){
        std::vector<ImplType> data;
        x.tensor2Array(data);
        this->data_.reset(new ImplType[data.size()]);   
        this->dim_ = x.getDim();
        this->offset_ = 0;
        this->current_dim = 0;
        this->shape_.reset(new int[this->dim_]);
        this->stride_.reset(new int[this->dim_]);
        for(int i = this->dim_ - 1; i >=0; i--){
            this->shape_.get()[i] = x.getShape(i);
            if(i == this->dim_ - 1)
                this->stride_.get()[i] = 1;
            else
                this->stride_.get()[i] = this->stride_.get()[i + 1] * this->shape_.get()[i + 1];
        }
        for(int i = 0; i < data.size(); i++)
            this->data_.get()[i] = data[i];
    }
    template<class ImplType, int N>
    Tensor<ImplType, N> :: Tensor(const Tensor<ImplType, N> &x){
        std::vector<ImplType> data;
        x.tensor2Array(data);
        this->data_.reset(new ImplType[data.size()]);   
        this->dim_ = x.getDim();
        this->offset_ = 0;
        this->current_dim = 0;
        this->shape_.reset(new int[this->dim_]);
        this->stride_.reset(new int[this->dim_]);
        for(int i = this->dim_ - 1; i >=0; i--){
            this->shape_.get()[i] = x.getShape(i);
            if(i == this->dim_ - 1)
                this->stride_.get()[i] = 1;
            else
                this->stride_.get()[i] = this->stride_.get()[i + 1] * this->shape_.get()[i + 1];
        }
        for(int i = 0; i < data.size(); i++)
            this->data_.get()[i] = data[i];
    }
    template<class ImplType, int N>
    Tensor<ImplType, N> :: Tensor(const std::vector<int> values, ImplType*& data){
        int ElementSize = 1;
        for(auto value : values)    ElementSize *= value;
        this->data_ = std::shared_ptr<ImplType>
            (new ImplType[ElementSize], std::default_delete<ImplType[]>());
        // std::cout<<"新版本已调用1"<<std::endl;
        takeOwnership(data);
        this->offset_ = 0;
        this->dim_ = values.size();
        this->shape_ = std::shared_ptr<int>(new int[this->dim_], std::default_delete<int[]>());
        this->stride_ = std::shared_ptr<int>(new int[this->dim_], std::default_delete<int[]>());
        auto it = values.end();
        it--;
        for(int idx = this->dim_ - 1; idx >= 0; idx--) {
            this->shape_.get()[idx] = *it;
            if(idx == this->dim_ - 1) 
                this->stride_.get()[idx] = 1;
            else 
                this->stride_.get()[idx] = this->stride_.get()[idx + 1] * this->shape_.get()[idx + 1];
            it--;
        }
    }
    template<class ImplType, int N>
    Tensor<ImplType, N> :: Tensor(const std::vector<int> values, std::vector<ImplType>& data){
        int ElementSize = 1;
        for(auto value : values)    ElementSize *= value;
        if(ElementSize != data.size())  
            throw std::runtime_error("The number of elements in the vector does not correspond to the Shape.");
        // this->data_ = std::shared_ptr<ImplType>
        //     (new ImplType[data.size()], std::default_delete<ImplType[]>());
        // for(size_t i = 0; i < data.size(); i++)
        //     this->data_.get()[i] = data[i];
        // std::cout<<"新版本已调用2"<<std::endl;
        takeOwnership( data);
        this->offset_ = 0;
        this->dim_ = values.size();
        this->shape_ = std::shared_ptr<int>(new int[this->dim_], std::default_delete<int[]>());
        this->stride_ = std::shared_ptr<int>(new int[this->dim_], std::default_delete<int[]>());
        auto it = values.end();
        it--;
        for(int idx = this->dim_ - 1; idx >= 0; idx--) {
            this->shape_.get()[idx] = *it;
            if(idx == this->dim_ - 1) 
                this->stride_.get()[idx] = 1;
            else 
                this->stride_.get()[idx] = this->stride_.get()[idx + 1] * this->shape_.get()[idx + 1];
            it--;
        }
    }
    template<class ImplType, int N>
    Tensor<ImplType, N> :: Tensor(const std::vector<int> values, ImplType data){
        int ElementSize = 1;
        for(auto value : values)    ElementSize *= value;
        this->data_ = std::shared_ptr<ImplType>
            (new ImplType[ElementSize], std::default_delete<ImplType[]>());
        for(int i = 0; i < ElementSize; i++)
            this->data_.get()[i] = data;
        this->offset_ = 0;
        this->dim_ = values.size();
        this->shape_ = std::shared_ptr<int>(new int[this->dim_], std::default_delete<int[]>());
        this->stride_ = std::shared_ptr<int>(new int[this->dim_], std::default_delete<int[]>());
        auto it = values.end();
        it--;
        for(int idx = this->dim_ - 1; idx >= 0; idx--) {
            this->shape_.get()[idx] = *it;
            if(idx == this->dim_ - 1) 
                this->stride_.get()[idx] = 1;
            else 
                this->stride_.get()[idx] = this->stride_.get()[idx + 1] * this->shape_.get()[idx + 1];
            it--;
        }
    }
    template<class ImplType, int N>
    Tensor<ImplType, N> :: Tensor(std::shared_ptr<ImplType> data, int offset, int dim, std::shared_ptr<int> shape, std::shared_ptr<int> stride) {
        this->data_ = data;
        this->offset_ = offset;
        this->dim_ = dim;
        this->shape_ = shape;
        this->stride_ = stride;
    }
    template<class ImplType, int N>
    Tensor<ImplType, N> :: Tensor(std::shared_ptr<ImplType> data, int offset, int dim, std::shared_ptr<int> shape, std::shared_ptr<int> stride, int currentdim) {
        this->data_ = data;
        this->offset_ = offset;
        this->dim_ = dim;
        this->shape_ = shape;
        this->stride_ = stride;
        this->current_dim = currentdim;
    }
    template<class ImplType, int N>
    Tensor<ImplType, N>& Tensor<ImplType, N> :: operator=(const Tensor<ImplType, N>& operand) {
        std::vector<ImplType> data;
        operand.tensor2Array(data);
        this->data_.reset(new ImplType[data.size()]);   
        this->offset_ = 0;
        this->current_dim = 0;
        for(int i = this->dim_ - 1; i >=0; i--){
            this->shape_.get()[i] = operand.getShape(i);
            if(i == this->dim_ - 1)
                this->stride_.get()[i] = 1;
            else
                this->stride_.get()[i] = this->stride_.get()[i + 1] * this->shape_.get()[i + 1];
        }
        for(int i = 0; i < data.size(); i++)
            this->data_.get()[i] = data[i];
        return *this;
    }
    template<class ImplType, int N>
    TensorProxy<ImplType, N> Tensor<ImplType, N> :: operator[](std::initializer_list<int> idx) {
        int start , end, stride = 1;
        if(idx.size() == 0){
            return TensorProxy<ImplType, N>(*this, this->current_dim, 0, this->shape_.get()[this->current_dim], 1, 1);
        }
        else if(idx.size() == 1){
            const int i = *(idx.begin());
            return TensorProxy<ImplType, N>(*this, this->current_dim, i, i + 1, 1, 1);
        }else {
            const int *i = idx.begin();
            start = *i;
            i++;
            end = *i;
            i++;
            if(i!=idx.end())    
                stride = *i;
            return TensorProxy<ImplType, N>(*this, this->current_dim, start, end, stride, 1);
        }
    }
    template<class ImplType, int N>
    TensorProxy<ImplType, N-1> Tensor<ImplType, N> :: operator[](int idx) const {
        return TensorProxy<ImplType, N - 1>(*this, this->current_dim, idx);
    }
    template<class ImplType, int N>
    TensorProxy<ImplType, N> Tensor<ImplType, N> :: operator[](split sp) const {
        return TensorProxy<ImplType, N>(*this, 0, 0, this->shape_.get()[this->current_dim], 1, 0);
    }
    template<class ImplType, int N>
    TensorProxy<ImplType, N - 1> Tensor<ImplType, N> :: operator[](index sp) const {
        return TensorProxy<ImplType, N-1>(*this, this->current_dim, 0);
    }
    template<class ImplType, int N>
    Tensor<ImplType, N-1> Tensor<ImplType, N> :: slice(int dimIdx, int idx) const {
        // 参数检查
        if(dimIdx >= this->dim_ || idx >= this->shape_.get()[dimIdx]) 
            throw std::runtime_error("[int] operates on dimensions that exceed those of Tensor.");
        int offset = this->offset_ + idx * this->stride_.get()[dimIdx];
        int dim = this->dim_ - 1;
        std::shared_ptr<int> shape(new int[dim], std::default_delete<int[]>());
        std::shared_ptr<int> stride(new int[dim], std::default_delete<int[]>());
        for(int idx = 0; idx < dim; idx++) {
            if(idx < dimIdx) {
                shape.get()[idx] = this->shape_.get()[idx];
                stride.get()[idx] = this->stride_.get()[idx];
            }
            else {
                shape.get()[idx] = this->shape_.get()[idx + 1];
                stride.get()[idx] = this->stride_.get()[idx + 1];
            }
        }
        return Tensor<ImplType, N - 1>(this->data_, offset, dim, shape, stride, this->current_dim);
    }
    template<class ImplType, int N>
    Tensor<ImplType, N> Tensor<ImplType, N> :: slice(int dimIdx, int start, int end, int sliceStride, int ModifyDim) const {
        // 参数检查
        if(dimIdx >= this->dim_ || start >= this->shape_.get()[dimIdx] || end > this->shape_.get()[dimIdx]
        || start < 0 || end < 0 || start > end) 
            throw std::runtime_error("[{}] operates on dimensions that exceed those of Tensor.");
        int offset = this->offset_ + start * this->stride_.get()[dimIdx];
        int dim = this->dim_;
        std::shared_ptr<int> shape(new int[dim], std::default_delete<int[]>());
        std::shared_ptr<int> stride(new int[dim], std::default_delete<int[]>());
        for(int i = 0; i < dim; i++) {
            shape.get()[i] = this->shape_.get()[i];
            stride.get()[i] = this->stride_.get()[i];
        }
        shape.get()[dimIdx] = (end - start - 1) / sliceStride + 1;
        stride.get()[dimIdx] = this->stride_.get()[dimIdx] * sliceStride;
        return Tensor<ImplType, N>(this->data_, offset, dim, shape, stride, this->current_dim + ModifyDim);
    }
    #undef Base

    template<class ImplType>
    class Tensor <ImplType, 1> : public TensorBase<ImplType>{
    private:
        template <class InputIt>
        void initialize(InputIt first, InputIt last);
    public:
        friend class dacpp::TensorProxy<ImplType, 1>;
        Tensor(){
            this->dim_ = 1;
            this->shape_.reset(new int[1]);
            this->stride_.reset(new int[1]);
        };
        Tensor(const TensorProxy<ImplType, 1> &x);
        Tensor(const TensorProxy<ImplType, 1> &&x){
            this->data_ = x.getDataPtr();
            this->offset_ = x.getOffset();
            this->dim_ = x.getDim();
            this->stride_ = x.getStridePtr();
            this->shape_ = x.getShapePtr();
            this->current_dim = 0;
        }
        Tensor(const Tensor<ImplType, 1> &x);
        Tensor(std::vector<ImplType> init);
        Tensor(int len, const ImplType *init);
        Tensor(std::shared_ptr<ImplType> data, int offset, int dim, std::shared_ptr<int> shape, std::shared_ptr<int> stride);
        Tensor(std::shared_ptr<ImplType> data, int offset, int dim, std::shared_ptr<int> shape, std::shared_ptr<int> stride, int currentdim);
        
        Tensor& operator=(const Tensor<ImplType, 1>& operand);
        Tensor& operator=(const TensorProxy<ImplType, 1>& operand){
            std::vector<ImplType> data;
            operand.tensor2Array(data);
            this->data_.reset(new ImplType[data.size()]);   
            this->dim_ = 1;
            this->offset_ = 0;
            this->current_dim = 0;
            this->shape_.reset(new int[this->dim_]);
            this->stride_.reset(new int[this->dim_]);
            for(int i = this->dim_ - 1; i >=0; i--){
                this->shape_.get()[i] = operand.getShape(i);
                if(i == this->dim_ - 1)
                    this->stride_.get()[i] = 1;
                else
                    this->stride_.get()[i] = this->stride_.get()[i + 1] * this->shape_.get()[i + 1];
            }
            for(int i = 0; i < data.size(); i++)
                this->data_.get()[i] = data[i];
            return *this;
        }
        Tensor& operator=(const TensorProxy<ImplType, 1>&& operand){
            this->data_ = operand.getDataPtr();
            this->offset_ = operand.getOffset();
            this->dim_ = operand.getDim();
            this->stride_ = operand.getStridePtr();
            this->shape_ = operand.getShapePtr();
            this->current_dim = 0;
            return *this;
        }
        TensorProxy<ImplType, 1> operator[](std::initializer_list<int> idx);
        ImplType& operator[](int idx);
        TensorProxy<ImplType, 1> operator[](split sp) const;
        ImplType& operator[](index sp) const;
        ImplType& slice(int dimIdx, int idx) const;
        Tensor<ImplType, 1> slice(int dimIdx, int start, int end, int sliceStride = 1 , int ModifyDim = 0) const;
        Tensor<ImplType, 1> operator+(const Tensor<ImplType, 1>& operand) const{};
        Tensor<ImplType, 1> operator-(const Tensor<ImplType, 1>& operand) const{};
        Tensor<ImplType, 1> operator*(const Tensor<ImplType, 1>& operand) const{};
        Tensor<ImplType, 1> operator/(const Tensor<ImplType, 1>& operand) const{};
        Tensor<ImplType, 1> operator%(const Tensor<ImplType, 1>& operand) const{};
        void operator+=(const Tensor<ImplType, 1>& operand){};
        void operator-=(const Tensor<ImplType, 1>& operand){};
        void operator*=(const Tensor<ImplType, 1>& operand){};
        void operator/=(const Tensor<ImplType, 1>& operand){};
        void operator%=(const Tensor<ImplType, 1>& operand){};
    };
#define Base Tensor<ImplType, 1>
    template<class ImplType>
    template <class InputIt>
    void Base :: initialize(InputIt first, InputIt last) {
        size_t size = std::distance(first, last);
        this->data_ = std::shared_ptr<ImplType>
            (new ImplType[size], std::default_delete<ImplType[]>());
        std::copy(first, last, this->data_.get());
        this->offset_ = 0;
        this->dim_ = 1;
        this->shape_ = std::shared_ptr<int>
            (new int[this->dim_], std::default_delete<int[]>());
        this->stride_ = std::shared_ptr<int>
            (new int[this->dim_], std::default_delete<int[]>());
        this->shape_.get()[0] = size;
        this->stride_.get()[0] = 1;
    }
    template<class ImplType>
    Base :: Tensor(const TensorProxy<ImplType, 1> &x){
        std::vector<ImplType> data;
        x.tensor2Array(data);
        this->data_.reset(new ImplType[data.size()]);   
        this->dim_ = x.getDim();
        this->offset_ = 0;
        this->current_dim = 0;
        this->shape_.reset(new int[this->dim_]);
        this->stride_.reset(new int[this->dim_]);
        for(int i = this->dim_ - 1; i >=0; i--){
            this->shape_.get()[i] = x.getShape(i);
            if(i == this->dim_ - 1)
                this->stride_.get()[i] = 1;
            else
                this->stride_.get()[i] = this->stride_.get()[i + 1] * this->shape_.get()[i + 1];
        }
        for(int i = 0; i < data.size(); i++)
            this->data_.get()[i] = data[i];
    }
    template<class ImplType>
    Base :: Tensor(const Tensor<ImplType, 1> &x){
        std::vector<ImplType> data;
        x.tensor2Array(data);
        this->data_.reset(new ImplType[data.size()]);   
        this->dim_ = x.getDim();
        this->offset_ = 0;
        this->current_dim = 0;
        this->shape_.reset(new int[this->dim_]);
        this->stride_.reset(new int[this->dim_]);
        for(int i = this->dim_ - 1; i >=0; i--){
            this->shape_.get()[i] = x.getShape(i);
            if(i == this->dim_ - 1)
                this->stride_.get()[i] = 1;
            else
                this->stride_.get()[i] = this->stride_.get()[i + 1] * this->shape_.get()[i + 1];
        }
        for(int i = 0; i < data.size(); i++)
            this->data_.get()[i] = data[i];
    }
    template<class ImplType>
    Base :: Tensor(std::vector<ImplType> init){
        initialize(init.begin(), init.end());
    }
    template<class ImplType>
    Base :: Tensor(int len, const ImplType *init){
        this->data_ = std::shared_ptr<ImplType>
            (new ImplType[len], std::default_delete<ImplType[]>());
        std::copy_n(init, len, this->data_.get());
        this->offset_ = 0;
        this->dim_ = 1;
        this->shape_ = std::shared_ptr<int>
            (new int[this->dim_], std::default_delete<int[]>());
        this->stride_ = std::shared_ptr<int>
            (new int[this->dim_], std::default_delete<int[]>());
        this->shape_.get()[0] = len;
        this->stride_.get()[0] = 1;
    }
    template<class ImplType>
    Base :: Tensor(std::shared_ptr<ImplType> data, int offset, int dim, std::shared_ptr<int> shape, std::shared_ptr<int> stride) {
        this->data_ = data;
        this->offset_ = offset;
        this->dim_ = dim;
        this->shape_ = shape;
        this->stride_ = stride;
    }
    template<class ImplType>
    Base :: Tensor(std::shared_ptr<ImplType> data, int offset, int dim, std::shared_ptr<int> shape, std::shared_ptr<int> stride, int currentdim) {
        this->data_ = data;
        this->offset_ = offset;
        this->dim_ = dim;
        this->shape_ = shape;
        this->stride_ = stride;
        this->current_dim = currentdim;
    }
    template<class ImplType>
    Tensor<ImplType, 1>& Base :: operator=(const Tensor<ImplType, 1>& operand) {
        std::vector<ImplType> data;
        operand.tensor2Array(data);
        this->data_.reset(new ImplType[data.size()]);   
        this->offset_ = 0;
        this->current_dim = 0;
        for(int i = 0; i < this->dim_; i++){
            this->shape_.get()[i] = operand.getShape(i);
            this->stride_.get()[i] = operand.getStride(i);
        }
        for(int i = 0; i < data.size(); i++)
            this->data_.get()[i] = data[i];
        return *this;
    }
    template<class ImplType>
    TensorProxy<ImplType, 1> Base :: operator[](std::initializer_list<int> idx) {
        int start , end, stride = 1;
        if(idx.size() == 0)
            return TensorProxy<ImplType, 1>(*this, this->current_dim, 0, this->shape_.get()[this->current_dim], 1, 1);
        else if(idx.size() == 1){
            const int i = *(idx.begin());
            return TensorProxy<ImplType, 1>(*this, this->current_dim, i, i + 1, 1, 1);
        }else {
            const int *i = idx.begin();
            start = *i;
            i++;
            end = *i;
            i++;
            if(i!=idx.end())    
                stride = *i;
            return TensorProxy<ImplType, 1>(*this, this->current_dim, start, end, stride, 1);
        }
    }
    template<class ImplType>
    ImplType& Base :: operator[](int idx) {return slice(this->current_dim, idx);}
    template<class ImplType>
    TensorProxy<ImplType, 1> Base :: operator[](split sp) const {return TensorProxy<ImplType, 1>(*this, 0, 0, this->shape_.get()[this->current_dim], 1, 0);}
    template<class ImplType>
    ImplType& Base :: operator[](index sp) const{return slice(this->current_dim, 0);}
    template<class ImplType>
    ImplType& Base :: slice(int dimIdx, int idx) const {
        if(dimIdx >= this->dim_ || idx >= this->shape_.get()[dimIdx]) 
            throw std::runtime_error("[int] operates on dimensions that exceed those of Tensor.");
        int offset = this->offset_ + idx * this->stride_.get()[dimIdx];
        return this->data_.get()[offset];
    }
    template<class ImplType>
    Tensor<ImplType, 1> Base :: slice(int dimIdx, int start, int end, int sliceStride, int ModifyDim) const {
        // 参数检查
        if(dimIdx >= this->dim_ || start >= this->shape_.get()[dimIdx] || end > this->shape_.get()[dimIdx]
        || start < 0 || end < 0 || start > end) 
            throw std::runtime_error("[{}] operates on dimensions that exceed those of Tensor.");
        int offset = start * this->stride_.get()[dimIdx];
        int dim = this->dim_;
        std::shared_ptr<int> shape(new int[dim], std::default_delete<int[]>());
        std::shared_ptr<int> stride(new int[dim], std::default_delete<int[]>());
        for(int i = 0; i < dim; i++) {
            shape.get()[i] = this->shape_.get()[i];
            stride.get()[i] = this->stride_.get()[i];
        }
        shape.get()[dimIdx] = (end - start - 1) / sliceStride + 1;
        stride.get()[dimIdx] = this->stride_.get()[dimIdx] * sliceStride;
        return Tensor<ImplType, 1>(this->data_, offset, dim, shape, stride, this->current_dim + ModifyDim);
    }
    #undef Base
    template<class ImplType, int N>
    class TensorProxy: public TensorBase <ImplType>{
        friend class dacpp::Tensor<ImplType, N>;
        friend class dacpp::Tensor<ImplType, 1>;
    public:
        TensorProxy(const dacpp::Tensor<ImplType, N + 1> &tensor, int dimIdx, int idx);
        TensorProxy(const dacpp::Tensor<ImplType, N> &tensor, int dimIdx, int start, int end, int sliceStride = 1 , int ModifyDim = 0);
        TensorProxy(std::shared_ptr<ImplType> data, int offset, int dim, std::shared_ptr<int> shape, std::shared_ptr<int> stride, int currentdim);
        TensorProxy<ImplType, N> operator[](std::initializer_list<int> idx);
        TensorProxy<ImplType, N-1> operator[](int idx) const;
        TensorProxy<ImplType, N> operator[](split sp) const;
        TensorProxy<ImplType, N - 1> operator[](index sp) const;
        TensorProxy<ImplType, N-1> Pslice(int dimIdx, int idx) const;
        TensorProxy<ImplType, N> Pslice(int dimIdx, int start, int end, int sliceStride = 1 , int ModifyDim = 0) const;
        TensorProxy(const TensorProxy&& x){
            this->data_ = x.getDataPtr();
            this->offset_ = x.getOffset();
            this->dim_ = x.getDim();
            this->shape_ = x.getShapePtr();
            this->stride_ = x.getStridePtr();
            this->current_dim = 0;
        };
        TensorProxy(const TensorProxy& x){
            std::vector<ImplType> data;
            x.tensor2Array(data);
            this->data_.reset(new ImplType[data.size()]);   
            this->dim_ = x.getDim();
            this->offset_ = 0;
            this->current_dim = 0;
            this->shape_.reset(new int[this->dim_]);
            this->stride_.reset(new int[this->dim_]);
            for(int i = this->dim_ - 1; i >=0; i--){
                this->shape_.get()[i] = x.getShape(i);
                if(i == this->dim_ - 1)
                    this->stride_.get()[i] = 1;
                else
                    this->stride_.get()[i] = this->stride_.get()[i + 1] * this->shape_.get()[i + 1];
            }
            for(int i = 0; i < data.size(); i++)
                this->data_.get()[i] = data[i];
        }
        Tensor<ImplType, N> copy(){
            Tensor<ImplType, N>tmp = new Tensor<ImplType, N>();
            tmp.offset_ = 0;
            std::vector<ImplType> dtmp;
            this->tensor2Array(dtmp);
            tmp.data_.reset(new ImplType[dtmp.size()]); 
            for(int i = tmp.dim_ - 1; i >=0; i--){
                tmp.shape_.get()[i] = this->getShape(i);
                if(i == tmp.dim_ - 1)
                    tmp.stride_.get()[i] = 1;
                else
                    tmp.stride_.get()[i] = tmp.stride_.get()[i + 1] * tmp.shape_.get()[i + 1];
            }
            for(int i = 0; i < dtmp.size(); i++)
                tmp.data_.get()[i] = dtmp[i];
            return tmp;
        }
        template<int M>
        TensorProxy<ImplType, N>& operator =(const TensorProxy<ImplType, M>& operand);
        template<int M>
        TensorProxy<ImplType, N>& operator =(const TensorProxy<ImplType, M>&& operand);
        template<int M>
        TensorProxy<ImplType, N>& operator =(const Tensor<ImplType, M>& operand);

        TensorProxy<ImplType, N> operator+(const TensorProxy<ImplType, N>& operand) const{};
        TensorProxy<ImplType, N> operator-(const TensorProxy<ImplType, N>& operand) const{};
        TensorProxy<ImplType, N> operator*(const TensorProxy<ImplType, N>& operand) const{};
        TensorProxy<ImplType, N> operator/(const TensorProxy<ImplType, N>& operand) const{};
        TensorProxy<ImplType, N> operator%(const TensorProxy<ImplType, N>& operand) const{};
        void operator+=(const TensorProxy<ImplType, N>& operand){};
        void operator-=(const TensorProxy<ImplType, N>& operand){};
        void operator*=(const TensorProxy<ImplType, N>& operand){};
        void operator/=(const TensorProxy<ImplType, N>& operand){};
        void operator%=(const TensorProxy<ImplType, N>& operand){};
    };
#define Base TensorProxy<ImplType, N>

    template<class ImplType, int N>
    Base :: TensorProxy(const dacpp::Tensor<ImplType, N + 1> &tensor, int dimIdx, int idx){
        if(dimIdx >= tensor.getDim() || idx >= tensor.getShape(dimIdx))
            throw std::runtime_error("[int] operates on dimensions that exceed those of Tensor.");
        int offset = tensor.getOffset() + idx * tensor.getStride(dimIdx);
        int dim = tensor.getDim() - 1;
        std::shared_ptr<int> shape(new int[dim], std::default_delete<int[]>());
        std::shared_ptr<int> stride(new int[dim], std::default_delete<int[]>());
        for(int idx = 0; idx < dim; idx++) {
            if(idx < dimIdx) {
                shape.get()[idx] = tensor.getShape(idx);
                stride.get()[idx] = tensor.getStride(idx);
            }
            else {
                shape.get()[idx] = tensor.getShape(idx + 1);
                stride.get()[idx] = tensor.getStride(idx + 1);
            }
        }
        this->data_ = tensor.getDataPtr();
        this->offset_ = offset;
        this->dim_ = dim;
        this->shape_ = shape;
        this->stride_ = stride;
        this->current_dim = tensor.getCurrentDim();
    }
    template<class ImplType, int N>
    Base :: TensorProxy(const dacpp::Tensor<ImplType, N> &tensor, int dimIdx, int start, int end, int sliceStride, int ModifyDim){

        if(dimIdx >= tensor.getDim() || start >= tensor.getShape(dimIdx) || end > tensor.getShape(dimIdx)
        || start < 0 || end < 0 || start > end) 
            throw std::runtime_error("[{}] operates on dimensions that exceed those of TensorProxy.");
        int offset = start * tensor.getStride(dimIdx);
        int dim = tensor.getDim();
        std::shared_ptr<int> shape(new int[dim], std::default_delete<int[]>());
        std::shared_ptr<int> stride(new int[dim], std::default_delete<int[]>());
        for(int i = 0; i < dim; i++) {
            shape.get()[i] = tensor.getShape(i);
            stride.get()[i] = tensor.getStride(i);
        }
        shape.get()[dimIdx] = (end - start - 1) / sliceStride + 1;
        stride.get()[dimIdx] = tensor.getStride(dimIdx) * sliceStride;
        this->data_ = tensor.getDataPtr();
        this->offset_ = offset;
        this->dim_ = dim;
        this->shape_ = shape;
        this->stride_ = stride;
        this->current_dim = tensor.getCurrentDim()+ ModifyDim;
    }
    template<class ImplType, int N>
    Base :: TensorProxy(std::shared_ptr<ImplType> data, int offset, int dim, std::shared_ptr<int> shape, std::shared_ptr<int> stride, int currentdim) {
        this->data_ = data;
        this->offset_ = offset;
        this->dim_ = dim;
        this->shape_ = shape;
        this->stride_ = stride;
        this->current_dim = currentdim;
    }
    template<class ImplType, int N>
    TensorProxy<ImplType, N> Base :: operator[](std::initializer_list<int> idx) {
        int start , end, stride = 1;
        if(idx.size() == 0)
            return Pslice(this->current_dim, 0, this->shape_.get()[this->current_dim], 1, 1);
        else if(idx.size() == 1){
            start = *(idx.begin());
            return Pslice(this->current_dim, start, start + 1, 1, 1);
        }else {
            const int *i = idx.begin();
            start = *i;
            i++;
            end = *i;
            i++;
            if(i!=idx.end())    
                stride = *i;
            return Pslice(this->current_dim, start, end, stride, 1);
        }
    }
    template<class ImplType, int N>
    TensorProxy<ImplType, N-1> Base :: operator[](int idx) const {
        return Pslice(this->current_dim, idx);
    }
    template<class ImplType, int N>
    TensorProxy<ImplType, N> Base :: operator[](split sp) const {
        return *this;
    }
    template<class ImplType, int N>
    TensorProxy<ImplType, N - 1> Base :: operator[](index sp) const {
        return Pslice(this->current_dim, 0);
    }
    template<class ImplType, int N>
    TensorProxy<ImplType, N-1> Base :: Pslice(int dimIdx, int idx) const {
        if(dimIdx >= this->dim_ || idx >= this->shape_.get()[dimIdx]) 
            throw std::runtime_error("[int] operates on dimensions that exceed those of TensorProxy.");
        int offset = this->offset_ + idx * this->stride_.get()[dimIdx];
        int dim = this->dim_ - 1;
        std::shared_ptr<int> shape(new int[dim], std::default_delete<int[]>());
        std::shared_ptr<int> stride(new int[dim], std::default_delete<int[]>());
        for(int idx = 0; idx < dim; idx++) {
            if(idx < dimIdx) {
                shape.get()[idx] = this->shape_.get()[idx];
                stride.get()[idx] = this->stride_.get()[idx];
            }
            else {
                shape.get()[idx] = this->shape_.get()[idx + 1];
                stride.get()[idx] = this->stride_.get()[idx + 1];
            }
        }
        return TensorProxy<ImplType, N - 1>(this->data_, offset, dim, shape, stride, this->current_dim);
    }
    template<class ImplType, int N>
    TensorProxy<ImplType, N> Base :: Pslice(int dimIdx, int start, int end, int sliceStride, int ModifyDim) const {
        if(dimIdx >= this->dim_ || start >= this->shape_.get()[dimIdx] || end > this->shape_.get()[dimIdx]
        || start < 0 || end < 0 || start > end) 
            throw std::runtime_error("[{}] operates on dimensions that exceed those of TensorProxy.");
        int offset = this->offset_ + start * this->stride_.get()[dimIdx];
        int dim = this->dim_;
        std::shared_ptr<int> shape(new int[dim], std::default_delete<int[]>());
        std::shared_ptr<int> stride(new int[dim], std::default_delete<int[]>());
        for(int i = 0; i < dim; i++) {
            shape.get()[i] = this->shape_.get()[i];
            stride.get()[i] = this->stride_.get()[i];
        }
        shape.get()[dimIdx] = (end - start - 1) / sliceStride + 1;
        stride.get()[dimIdx] = this->stride_.get()[dimIdx] * sliceStride;
        return TensorProxy<ImplType, N>(this->data_, offset, dim, shape, stride, this->current_dim + ModifyDim);
    }
    template<class ImplType, int N>
    template<int M>
    TensorProxy<ImplType, N>& TensorProxy<ImplType, N> :: operator =(const TensorProxy<ImplType, M>& operand){
        std::vector <ImplType> Dl, Dr;
        this->tensor2Array(Dl);
        operand.tensor2Array(Dr);
        if(Dl.size()!=Dr.size())
            throw std::runtime_error("The size of both tensor is not equal!");
        this->array2Tensor(Dr);
        return *this;
    }
    template<class ImplType, int N>
    template<int M>
    TensorProxy<ImplType, N>& TensorProxy<ImplType, N> :: operator =(const TensorProxy<ImplType, M>&& operand){
        std::vector <ImplType> Dl, Dr;
        this->tensor2Array(Dl);
        operand.tensor2Array(Dr);
        if(Dl.size()!=Dr.size())
            throw std::runtime_error("The size of both tensor is not equal!");
        this->array2Tensor(Dr);
        return *this;
    }
    template<class ImplType, int N>
    template<int M>
    TensorProxy<ImplType, N>& TensorProxy<ImplType, N> ::operator =(const Tensor<ImplType, M>& operand){
        std::vector <ImplType> Dl, Dr;
        this->tensor2Array(Dl);
        operand.tensor2Array(Dr);
        if(Dl.size()!=Dr.size())
            throw std::runtime_error("The size of both tensor is not equal!");
        this->array2Tensor(Dr);
        return *this;
    }
    #undef Base
    template<class ImplType>
    class TensorProxy <ImplType, 1> : public TensorBase<ImplType>{
        friend class dacpp::Tensor<ImplType, 2>;
        friend class dacpp::Tensor<ImplType, 1>;
    public:
        TensorProxy(const dacpp::Tensor<ImplType, 2> &tensor, int dimIdx, int idx);
        TensorProxy(const dacpp::Tensor<ImplType, 1> &tensor, int dimIdx, int start, int end, int sliceStride = 1 , int ModifyDim = 0);
        TensorProxy(std::shared_ptr<ImplType> data, int offset, int dim, std::shared_ptr<int> shape, std::shared_ptr<int> stride, int currentdim);
        TensorProxy<ImplType, 1> operator[](std::initializer_list<int> idx);
        ImplType& operator[](int idx);
        TensorProxy<ImplType, 1> operator[](split sp)const;
        ImplType& operator[](index sp) const;
        ImplType& Pslice(int dimIdx, int idx) const;
        TensorProxy<ImplType, 1> Pslice(int dimIdx, int start, int end, int sliceStride = 1 , int ModifyDim = 0) const;
        TensorProxy(const TensorProxy&& x){
            this->data_ = x.getDataPtr();
            this->offset_ = x.getOffset();
            this->dim_ = x.getDim();
            this->shape_ = x.getShapePtr();
            this->stride_ = x.getStridePtr();
            this->current_dim = 0;
        }
        TensorProxy(const TensorProxy& x){
            std::vector<ImplType> data;
            x.tensor2Array(data);
            this->data_.reset(new ImplType[data.size()]);   
            this->dim_ = x.getDim();
            this->offset_ = 0;
            this->current_dim = 0;
            this->shape_.reset(new int[this->dim_]);
            this->stride_.reset(new int[this->dim_]);
            for(int i = this->dim_ - 1; i >=0; i--){
                this->shape_.get()[i] = x.getShape(i);
                if(i == this->dim_ - 1)
                    this->stride_.get()[i] = 1;
                else
                    this->stride_.get()[i] = this->stride_.get()[i + 1] * this->shape_.get()[i + 1];
            }
            for(int i = 0; i < data.size(); i++)
                this->data_.get()[i] = data[i];
        }
        Tensor<ImplType, 1> copy(){
            Tensor<ImplType, 1>tmp = Tensor<ImplType, 1>();
            tmp.offset_ = 0;
            std::vector<ImplType> dtmp;
            this->tensor2Array(dtmp);
            tmp.data_.reset(new ImplType[dtmp.size()]); 
            for(int i = tmp.dim_ - 1; i >=0; i--){
                tmp.shape_.get()[i] = this->getShape(i);
                if(i == tmp.dim_ - 1)
                    tmp.stride_.get()[i] = 1;
                else
                    tmp.stride_.get()[i] = tmp.stride_.get()[i + 1] * tmp.shape_.get()[i + 1];
            }
            for(int i = 0; i < dtmp.size(); i++)
                tmp.data_.get()[i] = dtmp[i];
            return tmp;
        }
        template<int M>
        TensorProxy& operator =(const TensorProxy<ImplType, M>& operand);
        template<int M>
        TensorProxy& operator =(const TensorProxy<ImplType, M>&& operand);
        template<int M>
        TensorProxy& operator =(const Tensor<ImplType, M>& operand);

        TensorProxy<ImplType, 1> operator+(const TensorProxy<ImplType, 1>& operand) const{};
        TensorProxy<ImplType, 1> operator-(const TensorProxy<ImplType, 1>& operand) const{};
        TensorProxy<ImplType, 1> operator*(const TensorProxy<ImplType, 1>& operand) const{};
        TensorProxy<ImplType, 1> operator/(const TensorProxy<ImplType, 1>& operand) const{};
        TensorProxy<ImplType, 1> operator%(const TensorProxy<ImplType, 1>& operand) const{};
        void operator+=(const TensorProxy<ImplType, 1>& operand){};
        void operator-=(const TensorProxy<ImplType, 1>& operand){};
        void operator*=(const TensorProxy<ImplType, 1>& operand){};
        void operator/=(const TensorProxy<ImplType, 1>& operand){};
        void operator%=(const TensorProxy<ImplType, 1>& operand){};
    };
#define Base TensorProxy<ImplType, 1>

    template<class ImplType>
    Base :: TensorProxy(const dacpp::Tensor<ImplType, 2> &tensor, int dimIdx, int idx){
        if(dimIdx >= tensor.getDim() || idx >= tensor.getShape(dimIdx))
            throw std::runtime_error("[int] operates on dimensions that exceed those of Tensor.");
        int offset = tensor.getOffset() + idx * tensor.getStride(dimIdx);
        int dim = tensor.getDim() - 1;
        std::shared_ptr<int> shape(new int[dim], std::default_delete<int[]>());
        std::shared_ptr<int> stride(new int[dim], std::default_delete<int[]>());
        for(int idx = 0; idx < dim; idx++) {
            if(idx < dimIdx) {
                shape.get()[idx] = tensor.getShape(idx);
                stride.get()[idx] = tensor.getStride(idx);
            }
            else {
                shape.get()[idx] = tensor.getShape(idx + 1);
                stride.get()[idx] = tensor.getStride(idx + 1);
            }
        }
        this->data_ = tensor.getDataPtr();
        this->offset_ = offset;
        this->dim_ = dim;
        this->shape_ = shape;
        this->stride_ = stride;
        this->current_dim = tensor.getCurrentDim();
    }
    template<class ImplType>
    Base :: TensorProxy(const dacpp::Tensor<ImplType, 1> &tensor, int dimIdx, int start, int end, int sliceStride, int ModifyDim){
        if(dimIdx >= tensor.getDim() || start >= tensor.getShape(dimIdx) || end > tensor.getShape(dimIdx)
        || start < 0 || end < 0 || start > end) 
            throw std::runtime_error("[{}] operates on dimensions that exceed those of TensorProxy.");
        int offset = start * tensor.getStride(dimIdx);
        int dim = tensor.getDim();
        std::shared_ptr<int> shape(new int[dim], std::default_delete<int[]>());
        std::shared_ptr<int> stride(new int[dim], std::default_delete<int[]>());
        for(int i = 0; i < dim; i++) {
            shape.get()[i] = tensor.getShape(i);
            stride.get()[i] = tensor.getStride(i);
        }
        shape.get()[dimIdx] = (end - start - 1) / sliceStride + 1;
        stride.get()[dimIdx] = tensor.getStride(dimIdx) * sliceStride;
        this->data_ = tensor.getDataPtr();
        this->offset_ = offset;
        this->dim_ = dim;
        this->shape_ = shape;
        this->stride_ = stride;
        this->current_dim = tensor.getCurrentDim()+ ModifyDim;
    }
    template<class ImplType>
    Base :: TensorProxy(std::shared_ptr<ImplType> data, int offset, int dim, std::shared_ptr<int> shape, std::shared_ptr<int> stride, int currentdim) {
        this->data_ = data;
        this->offset_ = offset;
        this->dim_ = dim;
        this->shape_ = shape;
        this->stride_ = stride;
        this->current_dim = currentdim;
    }
    template<class ImplType>
    TensorProxy<ImplType, 1> Base :: operator[](std::initializer_list<int> idx) {
        int start , end, stride = 1;
        if(idx.size() == 0)
            return Pslice(this->current_dim, 0, this->shape_.get()[this->current_dim], 1, 1);
        else if(idx.size() == 1){
            const int i = *(idx.begin());
            return Pslice(this->current_dim, i, i + 1, 1, 1);
        }else {
            const int *i = idx.begin();
            start = *i;
            i++;
            end = *i;
            i++;
            if(i!=idx.end())    
                stride = *i;
            return Pslice(this->current_dim, start, end, stride, 1);
        }
    }
    template<class ImplType>
    ImplType& Base :: operator[](int idx) {
        return Pslice(this->current_dim, idx);
    }
    template<class ImplType>
    TensorProxy<ImplType, 1> Base :: operator[](split sp) const {
        return *this;
    }
    template<class ImplType>
    ImplType& Base :: operator[](index sp) const{
        return Pslice(this->current_dim, 0);
    }
    template<class ImplType>
    ImplType& Base ::Pslice(int dimIdx, int idx) const {
        if(dimIdx >= this->dim_ || idx >= this->shape_.get()[dimIdx]) 
            throw("Slice operates on dimensions that exceed those of TensorProxy.");
        int offset = this->offset_ + idx * this->stride_.get()[dimIdx];
        return this->data_.get()[offset];
    }
    template<class ImplType>
    TensorProxy<ImplType, 1> Base :: Pslice(int dimIdx, int start, int end, int sliceStride, int ModifyDim) const {
        if(dimIdx >= this->dim_ || start >= this->shape_.get()[dimIdx] || end > this->shape_.get()[dimIdx]
        || start < 0 || end < 0 || start > end) 
            throw("Slice operates on dimensions that exceed those of TensorProxy.");
        int offset = this->offset_ + start * this->stride_.get()[dimIdx];
        int dim = this->dim_;
        std::shared_ptr<int> shape(new int[dim], std::default_delete<int[]>());
        std::shared_ptr<int> stride(new int[dim], std::default_delete<int[]>());
        for(int i = 0; i < dim; i++) {
            shape.get()[i] = this->shape_.get()[i];
            stride.get()[i] = this->stride_.get()[i];
        }
        shape.get()[dimIdx] = (end - start - 1) / sliceStride + 1;
        stride.get()[dimIdx] = this->stride_.get()[dimIdx] * sliceStride;
        return TensorProxy<ImplType, 1>(this->data_, offset, dim, shape, stride, this->current_dim + ModifyDim);
    }
    template<class ImplType>
    template<int M>
    TensorProxy<ImplType, 1>& TensorProxy<ImplType, 1> :: operator =(const TensorProxy<ImplType, M>& operand){
        std::vector <ImplType> Dl, Dr;
        this->tensor2Array(Dl);
        operand.tensor2Array(Dr);
        if(Dl.size()!=Dr.size())
            throw std::runtime_error("The size of both tensor is not equal!");
        this->array2Tensor(Dr);
        return *this;
    }
    template<class ImplType>
    template<int M>
    TensorProxy<ImplType, 1>& TensorProxy<ImplType, 1> :: operator =(const TensorProxy<ImplType, M>&& operand){
        std::vector <ImplType> Dl, Dr;
        this->tensor2Array(Dl);
        operand.tensor2Array(Dr);
        if(Dl.size()!=Dr.size())
            throw std::runtime_error("The size of both tensor is not equal!");
        this->array2Tensor(Dr);
        return *this;
    }
    template<class ImplType>
    template<int M>
    TensorProxy<ImplType, 1>& TensorProxy<ImplType, 1> ::operator =(const Tensor<ImplType, M>& operand){
        std::vector <ImplType> Dl, Dr;
        this->tensor2Array(Dl);
        operand.tensor2Array(Dr);
        if(Dl.size()!=Dr.size())
            throw std::runtime_error("The size of both tensor is not equal!");
        this->array2Tensor(Dr);
        return *this;
    }
    #undef Base

} // namespace dacpp

namespace dacpp {
    template <class T>
    using Vector = Tensor<T, 1>;
    template <class T>
    using Matrix = Tensor<T, 2>;
}

#endif