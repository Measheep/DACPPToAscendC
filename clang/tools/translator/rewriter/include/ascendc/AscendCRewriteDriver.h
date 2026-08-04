#ifndef TRANSLATOR_REWRITER_ASCENDC_REWRITE_DRIVER_H
#define TRANSLATOR_REWRITER_ASCENDC_REWRITE_DRIVER_H

namespace clang {
class Rewriter;
}

namespace dacppTranslator {
class DacppFile;

namespace ascendc {

void RewriteAscendC(DacppFile* dacppFile, clang::Rewriter* rewriter);

} // namespace ascendc
} // namespace dacppTranslator

#endif // TRANSLATOR_REWRITER_ASCENDC_REWRITE_DRIVER_H
