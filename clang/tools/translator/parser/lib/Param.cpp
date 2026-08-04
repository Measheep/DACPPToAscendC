#include <string>
#include "clang/AST/PrettyPrinter.h"
#include "clang/AST/Type.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "clang/AST/TemplateName.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/Basic/LLVM.h"
#include "Param.h"

using namespace llvm;

using namespace llvm;

/*
    参数
*/
dacppTranslator::Param::Param() {
}

void dacppTranslator::Param::setRw(bool rw) {
    this->rw = rw;
}

bool dacppTranslator::Param::getRw() {
    return rw;
}

// When printing a reference, the referenced type might also be a reference.
// If so, we want to skip that before printing the inner type.
static clang::QualType skipTopLevelReferences(clang::QualType T) {
  if (auto *Ref = T->getAs<clang::ReferenceType>())
    return skipTopLevelReferences(Ref->getPointeeTypeAsWritten());
  return T;
}

static clang::QualType GetBaseType(clang::QualType T) {
  // FIXME: This should be on the Type class!
  clang::QualType BaseType = T;
  while (!BaseType->isSpecifierType()) {
    if (const clang::PointerType *PTy = BaseType->getAs<clang::PointerType>())
      BaseType = PTy->getPointeeType();
    else if (const clang::ObjCObjectPointerType *OPT =
                 BaseType->getAs<clang::ObjCObjectPointerType>())
      BaseType = OPT->getPointeeType();
    else if (const clang::BlockPointerType *BPy = BaseType->getAs<clang::BlockPointerType>())
      BaseType = BPy->getPointeeType();
    else if (const clang::ArrayType *ATy = dyn_cast<clang::ArrayType>(BaseType))
      BaseType = ATy->getElementType();
    else if (const clang::FunctionType *FTy = BaseType->getAs<clang::FunctionType>())
      BaseType = FTy->getReturnType();
    else if (const clang::VectorType *VTy = BaseType->getAs<clang::VectorType>())
      BaseType = VTy->getElementType();
    else if (const clang::ReferenceType *RTy = BaseType->getAs<clang::ReferenceType>())
      BaseType = RTy->getPointeeType();
    else if (const clang::AutoType *ATy = BaseType->getAs<clang::AutoType>())
      BaseType = ATy->getDeducedType();
    else if (const clang::ParenType *PTy = BaseType->getAs<clang::ParenType>())
      BaseType = PTy->desugar();
    else
      // This must be a syntax error.
      break;
  }
  return BaseType;
}

void dacppTranslator::Param::setType(clang::QualType newType) {
  this->newType = newType;
  clang::SplitQualType split;
  split = newType.split();
  const clang::Type *ty = split.Ty;
  bool found_p = false;
  const clang::LValueReferenceType *LRT;
  clang::QualType Inner;
  clang::SplitQualType Split;
  const clang::Type *Ty;
  const clang::ElaboratedType *ET;
  clang::NestedNameSpecifier *Qualifier;
  const clang::TemplateSpecializationType *SpecTy;
  std::string BSBuf;
  llvm::raw_string_ostream BSStream(BSBuf);
  llvm::ArrayRef<clang::TemplateArgument> Args;

  LRT = dyn_cast<clang::LValueReferenceType>(ty);
  if (!LRT)
    goto fail;

  Inner = skipTopLevelReferences(LRT->getPointeeTypeAsWritten());
  Split = Inner.split();
  Ty = Inner.getTypePtrOrNull();
  if (Ty->getAsTagDecl()) {
    for (const clang::DeclContext *DC = Ty->getAsTagDecl()->getDeclContext();
         DC->isNamespace(); DC = DC->getParent()) {
      if (const auto *Namespace = dyn_cast<clang::NamespaceDecl>(DC)) {
        if (Namespace->getDeclName() &&
            strcmp(Namespace->getName().str().c_str(), "dacpp") == 0) {
          found_p = true;
        }
      }
    }
    if (!found_p)
      goto fail;
  }
  found_p = false;

  ET = dyn_cast<clang::ElaboratedType>(Split.Ty);
  if (!ET)
    goto fail;

  Qualifier = ET->getQualifier();
  if (Qualifier &&
      Qualifier->getKind() == clang::NestedNameSpecifier::Namespace &&
      strcmp(Qualifier->getAsNamespace()->getNameAsString().c_str(), "dacpp"))
    goto fail;

  if (ET->getOwnedTagDecl())
    goto fail;
  Split = ET->getNamedType().split();

  SpecTy = dyn_cast<clang::TemplateSpecializationType>(Split.Ty);
  if (!SpecTy)
    goto fail;

  SpecTy->getTemplateName().print(BSStream, clang::LangOptions(),
                                  clang::TemplateName::Qualified::None);
  if (strcmp("Tensor", BSBuf.c_str()) && strcmp("Vector", BSBuf.c_str()) &&
      strcmp("Matrix", BSBuf.c_str()))
    goto fail;

  found_p = true;
  Args = SpecTy->template_arguments();
  for (const auto &Arg : Args) {
    // Print the argument into a string.
    llvm::SmallString<128> Buf;
    llvm::raw_svector_ostream ArgOS(Buf);
    const clang::TemplateArgument &Argument = (Arg);
    if (clang::TemplateArgument::Type == Argument.getKind()) {
      newType = Argument.getAsType();
      break;
    }
    llvm_unreachable("unreachable");
  }

fail:
  if (!found_p)
    newType = GetBaseType(newType);
  this->BasicType = newType;
}

std::string dacppTranslator::Param::getType() {
    return newType.getAsString();
}

std::string dacppTranslator::Param::getBasicType() {
    return BasicType.getAsString();
}

void dacppTranslator::Param::setName(std::string name) {
    this->name = name;
}

std::string dacppTranslator::Param::getName() {
    return name;
}

void dacppTranslator::Param::setShape(int size) {
    shape.push_back(size);
}

void dacppTranslator::Param::setShape(int idx, int size) {
    shape[idx] = size;
}

int dacppTranslator::Param::getShape(int idx) {
    return shape[idx];
}

int dacppTranslator::Param::getDim() {
    return shape.size(); 
}

/*
    划分结构参数
*/
dacppTranslator::ShellParam::ShellParam() {
}

void dacppTranslator::ShellParam::setSplit(Split* split) {
    splits.push_back(split);
}

dacppTranslator::Split* dacppTranslator::ShellParam::getSplit(int idx) {
    return splits[idx];
}

int dacppTranslator::ShellParam::getNumSplit() {
    return splits.size();
}