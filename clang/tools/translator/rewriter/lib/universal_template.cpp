#include"universal_template.h"

namespace UNIVERSAL_TEMPLATE{

/*
	文本替换。
	将 text 中的 find 换成 replace。
*/
void replaceTextInString(std::string& text, 
    const std::string &find, 
    const std::string &replace){
	std::string::size_type pos = 0;
	while ((pos = text.find(find, pos)) != std::string::npos){
		text.replace(pos, find.length(), replace);
		pos += replace.length();
	}
}
/*
	文本替换。
	对 templ 做 replacements 中的文本替换。
*/
std::string templateString(std::string templ, 
    std::vector<std::pair<std::string, std::string>> replacements){
	for(auto &element : replacements)
		replaceTextInString(templ, element.first, element.second);
	return templ;
}

}
