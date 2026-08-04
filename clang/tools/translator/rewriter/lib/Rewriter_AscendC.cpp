#include "Rewriter.h"
#include "ascendc/AscendCRewriteDriver.h"

void dacppTranslator::Rewriter::rewriteDac_AscendC() {
    dacppTranslator::ascendc::RewriteAscendC(dacppFile, rewriter);
}
