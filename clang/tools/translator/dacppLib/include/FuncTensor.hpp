#ifndef FUNCTENSOR_H
#define FUNCTENSOR_H

#include<memory>
#include<vector>

using std::vector;

namespace dacpp{
    template<class T>
    class FuncTensor{
    public:
        FuncTensor(){};
        ~FuncTensor(){};
        FuncTensor(std::shared_ptr<T>data, int offset, int dim, std::shared_ptr<T>shape, std::shared_ptr<T>stride){
            this->offset_ = offset;
            this->dim_ = dim;
            int Element = 1;
            for(int i = 0; i < dim; i++){
                this->shape_.push_back(shape.get()[i]);
                this->stride_.push_back(stride.get()[i]);
                Element *= shape.get()[i];
            }
            recursiveTake(0, data);
        }

        std::vector<T> getData(){
            return this->data_;
        }
        constexpr int getOffset() const{
            return this->offset_;
        }
        constexpr int getDim() const {
            return this->dim_;
        }
        std::vector<int> getShape(){
            return this->shape_;
        }
        std::vector<int> getStride(){
            return this->stride_;
        }
    protected:
        void recursiveTake(int dimIdx, std::shared_ptr<T>_data_) {
            static vector<int> indices;
            if(dimIdx == dim_) {
                int index = offset_;
                for(int i = 0; i < dim_; i++)
                    index += indices[i] * stride_[i];
                this->data_.push_back(_data_.get()[index]);
                return;
            }
            for(int i = 0; i < shape_[dimIdx]; i++) {
                indices.push_back(i);
                recursiveTake(dimIdx + 1, _data_);
                indices.pop_back();
            }
        }
        std::vector<T> data_;
        int offset_;
        int dim_;
        std::vector<int> shape_;
        std::vector<int> stride_;
    };


}

#endif