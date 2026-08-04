#ifndef EXCEPTION_H
#define EXCEPTION_H

#include <exception>
#include <cstdio>
namespace dacpp {

struct Error: public std::exception {
    Error(const char* file, const char* func, unsigned int line);
    const char* what() const noexcept;

    static char msg_[300];
    const char* file_;
    const char* func_;
    const unsigned int line_;
};

}

#define ERROR_LOCATION __FILE__, __func__, __LINE__
#define THROW_ERROR(format, ...)    std::sprintf(::dacpp::Error::msg_, format, ##__VA_ARGS__);    \
    throw ::dacpp::Error(ERROR_LOCATION)

#endif