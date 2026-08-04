#ifndef UNIVERSITY_TEMPLATE_H
#define UNIVERSITY_TEMPLATE_H

#include<string>
#include<utility>
#include<vector>

namespace UNIVERSAL_TEMPLATE{

    void replaceTextInString(std::string& text, const std::string &find, const std::string &replace);

    std::string templateString(std::string templ, std::vector<std::pair<std::string, std::string>> replacements);

}

#endif
