#include "ascendc/AscendCSourceUtils.h"

#include <cmath>
#include <cctype>
#include <cstdlib>
#include <regex>
#include <unordered_map>
#include <vector>

#include "ascendc_mlir/CalcContext.h"

namespace dacppTranslator {
namespace ascendc {

std::string TrimCopy(const std::string& text) {
    size_t begin = text.find_first_not_of(" \t\n\r");
    if (begin == std::string::npos) {
        return "";
    }
    size_t end = text.find_last_not_of(" \t\n\r");
    return text.substr(begin, end - begin + 1);
}

std::string ReplaceAllRegex(const std::string& input,
                            const std::string& pattern,
                            const std::string& replacement) {
    try {
        return std::regex_replace(input, std::regex(pattern), replacement);
    } catch (...) {
        return input;
    }
}

std::string EscapeRegexText(const std::string& text) {
    std::string escaped;
    escaped.reserve(text.size() * 2);
    for (char c : text) {
        switch (c) {
        case '\\':
        case '^':
        case '$':
        case '.':
        case '|':
        case '?':
        case '*':
        case '+':
        case '(':
        case ')':
        case '[':
        case ']':
        case '{':
        case '}':
            escaped.push_back('\\');
            break;
        default:
            break;
        }
        escaped.push_back(c);
    }
    return escaped;
}

bool IsSimpleIdentifierText(const std::string& text) {
    return std::regex_match(TrimCopy(text), std::regex(R"(^[A-Za-z_]\w*$)"));
}

bool IsIdentCharText(char c) {
    return std::isalnum(static_cast<unsigned char>(c)) || c == '_';
}

std::string StripTrailingSemicolon(std::string text) {
    text = TrimCopy(text);
    while (!text.empty() && text.back() == ';') {
        text.pop_back();
        text = TrimCopy(text);
    }
    return text;
}

bool TryParseDoubleValue(const std::string& text, double& outValue) {
    std::string trimmed = TrimCopy(text);
    if (trimmed.empty()) {
        return false;
    }

    try {
        size_t idx = 0;
        outValue = std::stod(trimmed, &idx);
        while (idx < trimmed.size() && std::isspace(static_cast<unsigned char>(trimmed[idx]))) {
            ++idx;
        }
        return idx == trimmed.size();
    } catch (...) {
        return false;
    }
}

bool TryResolveSimpleSymbolValue(const std::string& sourceText,
                                 const std::string& symbol,
                                 double& outValue) {
    {
        dacppTranslator::ascendc_mlir::CalcContextSummary contextSummary =
            dacppTranslator::ascendc_mlir::ExtractCalcContext(nullptr, sourceText);
        if (dacppTranslator::ascendc_mlir::TryEvaluateConstExpr(
                TrimCopy(symbol), contextSummary, outValue)) {
            return true;
        }
    }

    auto SplitTopLevelByComma = [&](const std::string& text) {
        std::vector<std::string> parts;
        std::string current;
        int parenDepth = 0;
        int bracketDepth = 0;
        int braceDepth = 0;
        bool inSingleQuote = false;
        bool inDoubleQuote = false;
        bool escaping = false;

        auto FlushCurrent = [&]() {
            std::string trimmed = TrimCopy(current);
            if (!trimmed.empty()) {
                parts.push_back(trimmed);
            }
            current.clear();
        };

        for (char c : text) {
            if (escaping) {
                current += c;
                escaping = false;
                continue;
            }
            if (c == '\\') {
                current += c;
                escaping = true;
                continue;
            }
            if (!inDoubleQuote && c == '\'') {
                current += c;
                inSingleQuote = !inSingleQuote;
                continue;
            }
            if (!inSingleQuote && c == '"') {
                current += c;
                inDoubleQuote = !inDoubleQuote;
                continue;
            }
            if (!inSingleQuote && !inDoubleQuote) {
                if (c == '(') ++parenDepth;
                else if (c == ')') --parenDepth;
                else if (c == '[') ++bracketDepth;
                else if (c == ']') --bracketDepth;
                else if (c == '{') ++braceDepth;
                else if (c == '}') --braceDepth;
                else if (c == ',' && parenDepth == 0 && bracketDepth == 0 && braceDepth == 0) {
                    FlushCurrent();
                    continue;
                }
            }
            current += c;
        }

        FlushCurrent();
        return parts;
    };

    auto StripStaticCasts = [&](const std::string& expr) {
        std::string result = expr;
        std::regex castRegex(R"(static_cast<[^>]+>\s*\(\s*([^)]+?)\s*\))");
        for (int pass = 0; pass < 8; ++pass) {
            std::string replaced = std::regex_replace(result, castRegex, "$1");
            if (replaced == result) {
                break;
            }
            result = replaced;
        }
        return result;
    };

    struct NumericExprParser {
        const std::string& expr;
        const std::unordered_map<std::string, double>& symbols;
        const std::unordered_map<std::string, double>& locals;
        size_t pos = 0;
        bool ok = true;

        NumericExprParser(const std::string& e,
                          const std::unordered_map<std::string, double>& s,
                          const std::unordered_map<std::string, double>& l)
            : expr(e), symbols(s), locals(l) {}

        void SkipSpaces() {
            while (pos < expr.size() && std::isspace(static_cast<unsigned char>(expr[pos]))) {
                ++pos;
            }
        }

        std::string ParseIdentifier() {
            size_t begin = pos;
            if (begin >= expr.size() ||
                (!std::isalpha(static_cast<unsigned char>(expr[begin])) && expr[begin] != '_')) {
                ok = false;
                return "";
            }
            ++pos;
            while (pos < expr.size() &&
                   (std::isalnum(static_cast<unsigned char>(expr[pos])) || expr[pos] == '_')) {
                ++pos;
            }
            return expr.substr(begin, pos - begin);
        }

        double ParseNumber() {
            const char* beginPtr = expr.c_str() + pos;
            char* endPtr = nullptr;
            double value = std::strtod(beginPtr, &endPtr);
            if (endPtr == beginPtr) {
                ok = false;
                return 0.0;
            }

            pos += static_cast<size_t>(endPtr - beginPtr);
            while (pos < expr.size() && std::isalpha(static_cast<unsigned char>(expr[pos]))) {
                ++pos;
            }
            return value;
        }

        double ParseFactor() {
            SkipSpaces();
            if (!ok || pos >= expr.size()) {
                ok = false;
                return 0.0;
            }

            if (expr[pos] == '+') {
                ++pos;
                return ParseFactor();
            }
            if (expr[pos] == '-') {
                ++pos;
                return -ParseFactor();
            }

            if (expr[pos] == '(') {
                ++pos;
                double v = ParseExpression();
                SkipSpaces();
                if (pos >= expr.size() || expr[pos] != ')') {
                    ok = false;
                    return 0.0;
                }
                ++pos;
                return v;
            }

            if (std::isdigit(static_cast<unsigned char>(expr[pos])) || expr[pos] == '.') {
                return ParseNumber();
            }

            if (std::isalpha(static_cast<unsigned char>(expr[pos])) || expr[pos] == '_') {
                std::string ident = ParseIdentifier();
                if (!ok) {
                    return 0.0;
                }
                auto localIt = locals.find(ident);
                if (localIt != locals.end()) {
                    return localIt->second;
                }
                auto symIt = symbols.find(ident);
                if (symIt != symbols.end()) {
                    return symIt->second;
                }
                if (ident == "M_PI") {
                    return 3.14159265358979323846;
                }
                if (ident == "M_E") {
                    return 2.71828182845904523536;
                }
                ok = false;
                return 0.0;
            }

            ok = false;
            return 0.0;
        }

        double ParseTerm() {
            double lhs = ParseFactor();
            while (ok) {
                SkipSpaces();
                if (pos >= expr.size() || (expr[pos] != '*' && expr[pos] != '/')) {
                    break;
                }
                char op = expr[pos++];
                double rhs = ParseFactor();
                if (!ok) {
                    return 0.0;
                }
                if (op == '*') {
                    lhs *= rhs;
                } else {
                    if (std::fabs(rhs) < 1e-15) {
                        ok = false;
                        return 0.0;
                    }
                    lhs /= rhs;
                }
            }
            return lhs;
        }

        double ParseExpression() {
            double lhs = ParseTerm();
            while (ok) {
                SkipSpaces();
                if (pos >= expr.size() || (expr[pos] != '+' && expr[pos] != '-')) {
                    break;
                }
                char op = expr[pos++];
                double rhs = ParseTerm();
                if (!ok) {
                    return 0.0;
                }
                if (op == '+') {
                    lhs += rhs;
                } else {
                    lhs -= rhs;
                }
            }
            return lhs;
        }
    };

    auto TryEvalExpr = [&](const std::string& rawExpr,
                           const std::unordered_map<std::string, double>& symbols,
                           const std::unordered_map<std::string, double>& locals,
                           double& outNumeric) {
        std::string exprText = StripStaticCasts(TrimCopy(rawExpr));
        if (exprText.empty()) {
            return false;
        }
        NumericExprParser parser(exprText, symbols, locals);
        double value = parser.ParseExpression();
        parser.SkipSpaces();
        if (!parser.ok || parser.pos != exprText.size() || !std::isfinite(value)) {
            return false;
        }
        outNumeric = value;
        return true;
    };

    auto BuildNumericSymbols = [&](const std::string& text) {
        std::unordered_map<std::string, double> symbols;
        std::vector<std::pair<std::string, std::string>> pending;

        std::regex declStmtRegex(
            R"((?:^|[\n;])\s*(?:const\s+)?(?:size_t|int|long|float|double|uint32_t|uint64_t|std::size_t)\s+([^;]+);)");

        for (std::sregex_iterator it(text.begin(), text.end(), declStmtRegex), end; it != end; ++it) {
            std::string declList = (*it)[1].str();
            auto parts = SplitTopLevelByComma(declList);
            for (const std::string& part : parts) {
                std::smatch kv;
                std::regex kvRegex(R"(^\s*([A-Za-z_]\w*)\s*=\s*(.+?)\s*$)");
                if (std::regex_match(part, kv, kvRegex)) {
                    pending.emplace_back(kv[1].str(), TrimCopy(kv[2].str()));
                }
            }
        }

        for (size_t pass = 0; pass < pending.size() + 8; ++pass) {
            bool progressed = false;
            for (const auto& item : pending) {
                if (symbols.count(item.first) > 0) {
                    continue;
                }
                double value = 0.0;
                if (TryEvalExpr(item.second, symbols, {}, value)) {
                    symbols[item.first] = value;
                    progressed = true;
                }
            }
            if (!progressed) {
                break;
            }
        }

        return symbols;
    };

    double parsed = 0.0;
    if (TryParseDoubleValue(symbol, parsed)) {
        outValue = parsed;
        return true;
    }

    std::unordered_map<std::string, double> symbols = BuildNumericSymbols(sourceText);

    std::regex identRegex(R"(^[A-Za-z_]\w*$)");
    std::string trimmed = TrimCopy(symbol);
    if (std::regex_match(trimmed, identRegex)) {
        auto it = symbols.find(trimmed);
        if (it != symbols.end()) {
            outValue = it->second;
            return true;
        }
    }

    if (TryEvalExpr(trimmed, symbols, {}, outValue)) {
        return true;
    }

    return false;
}

} // namespace ascendc
} // namespace dacppTranslator
