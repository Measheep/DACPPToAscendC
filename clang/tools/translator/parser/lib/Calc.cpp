#include <string>
#include <regex>

#include "clang/AST/Attr.h"
#include "clang/AST/DeclFriend.h"
#include "clang/AST/DeclOpenMP.h"
#include "clang/Basic/Module.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/ADT/StringExtras.h"


#include "ASTParse.h"
#include "Calc.h"
#include "DacppStructure.h"
#include "Param.h"
#include "Shell.h"

#include <string>
#include <iostream>

namespace {


namespace declvisitor {
/// A simple visitor class that helps create declaration visitors.
template<template <typename> class Ptr, typename ImplClass, typename RetTy=void>
class Base {
public:
#define PTR(CLASS) typename Ptr<CLASS>::type
#define DISPATCH(NAME, CLASS) \
  return static_cast<ImplClass*>(this)->Visit##NAME(static_cast<PTR(CLASS)>(D))

  RetTy Visit(PTR(Decl) D) {
    switch (D->getKind()) {
#define DECL(DERIVED, BASE) \
      case Decl::DERIVED: DISPATCH(DERIVED##Decl, DERIVED##Decl);
#define ABSTRACT_DECL(DECL)
#include "clang/AST/DeclNodes.inc"
    }
    llvm_unreachable("Decl that isn't part of DeclNodes.inc!");
  }

  // 如果无法实现某种逻辑，就fall back
#define DECL(DERIVED, BASE) \
  RetTy Visit##DERIVED##Decl(PTR(DERIVED##Decl) D) { DISPATCH(BASE, BASE); }
#include "clang/AST/DeclNodes.inc"

  RetTy VisitDecl(PTR(Decl) D) { return RetTy(); }

#undef PTR
#undef DISPATCH
};

} // namespace declvisitor

// 一个帮助创建声明访问者的简单访问者类。
// 此类不保留Decl指针的一致性（另见ConstDeclVisitor）。
template <typename ImplClass, typename RetTy = void>
class DeclVisitor
    : public declvisitor::Base<std::add_pointer, ImplClass, RetTy> {};

// 此类保留了Decl指针的一致性
template <typename ImplClass, typename RetTy = void>
class ConstDeclVisitor
    : public declvisitor::Base<llvm::make_const_ptr, ImplClass, RetTy> {};

  class DeclPrinter : public DeclVisitor<DeclPrinter>, public StmtVisitor<DeclPrinter> {
    PrintingPolicy Policy;
    const ASTContext &Context;
    unsigned Indentation;
    bool PrintInstantiation;

    PrinterHelper* Helper;
    std::string NL;
    dacppTranslator::Calc *calc;

    raw_ostream& Indent() { return Indent(Indentation); }
    raw_ostream& Indent(unsigned Indentation);
    void ProcessDeclGroup(SmallVectorImpl<Decl*>& Decls);

    void Print(AccessSpecifier AS);
    void PrintConstructorInitializers(CXXConstructorDecl *CDecl,
                                      std::string &Proto);

    /// Print an Objective-C method type in parentheses.
    ///
    /// \param Quals The Objective-C declaration qualifiers.
    /// \param T The type to print.
    void PrintObjCMethodType(ASTContext &Ctx, Decl::ObjCDeclQualifier Quals,
                             QualType T);

    void PrintObjCTypeParams(ObjCTypeParamList *Params);

  public:
    raw_ostream &Out;

    virtual void PrintStmt(Stmt *S) { PrintStmt(S, Policy.Indentation); }

    virtual void PrintStmt(Stmt *S, int SubIndent) {
      Indentation += SubIndent;
      if (isa_and_nonnull<Expr>(S)) {
        // 如果这是stmt上下文中使用的expr，缩进并换行。
        Indent();
        Visit(S);
        Out << ";" << NL;
      } else if (S) {
        Visit(S);
      } else {
        Indent() << "<<<NULL STATEMENT>>>" << NL;
      }
      Indentation -= SubIndent;
    }

    virtual void PrintInitStmt(Stmt *S, unsigned PrefixWidth) {
      // 更好地应对奇数前缀宽度。
      Indentation += (PrefixWidth + 1) / 2;
      if (auto *DS = dyn_cast<DeclStmt>(S))
        PrintRawDeclStmt(DS);
      else
        PrintExpr(cast<Expr>(S));
      Out << "; ";
      Indentation -= (PrefixWidth + 1) / 2;
    }

    virtual void PrintControlledStmt(Stmt *S) {
      if (auto *CS = dyn_cast<CompoundStmt>(S)) {
        Out << " ";
        PrintRawCompoundStmt(CS);
        Out << NL;
      } else {
        Out << NL;
        PrintStmt(S);
      }
    }

    virtual void PrintRawCompoundStmt(CompoundStmt *S);
    virtual void PrintRawDecl(Decl *D);
    virtual void PrintRawDeclStmt(const DeclStmt *S);
    virtual void PrintRawIfStmt(IfStmt *If);
    virtual void PrintRawCXXCatchStmt(CXXCatchStmt *Catch);
    virtual void PrintCallArgs(CallExpr *E);
    virtual void PrintRawSEHExceptHandler(SEHExceptStmt *S);
    virtual void PrintRawSEHFinallyStmt(SEHFinallyStmt *S);
    virtual void PrintOMPExecutableDirective(OMPExecutableDirective *S,
                                     bool ForceNoStmt = false);
    virtual void PrintFPPragmas(CompoundStmt *S);

    virtual void PrintExpr(Expr *E) {
      if (E)
        Visit(E);
      else
        Out << "<null expr>";
    }

    virtual void Visit(Stmt* S) {
      if (Helper && Helper->handledStmt(S,Out))
          return;
      StmtVisitor<DeclPrinter>::Visit(S);
    }

    virtual void VisitStmt(Stmt *Node) LLVM_ATTRIBUTE_UNUSED {
      Indent() << "<<unknown stmt type>>" << NL;
    }

    virtual void VisitExpr(Expr *Node) LLVM_ATTRIBUTE_UNUSED {
      Out << "<<unknown expr type>>";
    }

    virtual void VisitCXXNamedCastExpr(CXXNamedCastExpr *Node);

#define ABSTRACT_STMT(CLASS)
#define STMT(CLASS, PARENT) \
    virtual void Visit##CLASS(CLASS *Node);
#include "clang/AST/StmtNodes.inc"

  virtual ~DeclPrinter() = default;

    DeclPrinter(dacppTranslator::Calc *calc, raw_ostream &Out, PrinterHelper *helper, const PrintingPolicy &Policy,
                const ASTContext &Context, unsigned Indentation = 0,StringRef NL = "\n", 
                bool PrintInstantiation = false)
        : Policy(Policy), Context(Context), Indentation(Indentation),
          PrintInstantiation(PrintInstantiation), Helper(helper), NL(NL), calc(calc) , Out(Out){}

    virtual void VisitDeclContext(DeclContext *DC, bool Indent = true);

    virtual void VisitTranslationUnitDecl(TranslationUnitDecl *D);
    virtual void VisitTypedefDecl(TypedefDecl *D);
    virtual void VisitTypeAliasDecl(TypeAliasDecl *D);
    virtual void VisitEnumDecl(EnumDecl *D);
    virtual void VisitRecordDecl(RecordDecl *D);
    virtual void VisitEnumConstantDecl(EnumConstantDecl *D);
    virtual void VisitEmptyDecl(EmptyDecl *D);
    virtual void VisitFunctionDecl(FunctionDecl *D);
    virtual void VisitFriendDecl(FriendDecl *D);
    virtual void VisitFieldDecl(FieldDecl *D);
    virtual void VisitVarDecl(VarDecl *D);
    virtual void VisitLabelDecl(LabelDecl *D);
    virtual void VisitParmVarDecl(ParmVarDecl *D);
    virtual void VisitFileScopeAsmDecl(FileScopeAsmDecl *D);
    virtual void VisitTopLevelStmtDecl(TopLevelStmtDecl *D);
    virtual void VisitImportDecl(ImportDecl *D);
    virtual void VisitStaticAssertDecl(StaticAssertDecl *D);
    virtual void VisitNamespaceDecl(NamespaceDecl *D);
    virtual void VisitUsingDirectiveDecl(UsingDirectiveDecl *D);
    virtual void VisitNamespaceAliasDecl(NamespaceAliasDecl *D);
    virtual void VisitCXXRecordDecl(CXXRecordDecl *D);
    virtual void VisitLinkageSpecDecl(LinkageSpecDecl *D);
    virtual void VisitTemplateDecl(const TemplateDecl *D);
    virtual void VisitFunctionTemplateDecl(FunctionTemplateDecl *D);
    virtual void VisitClassTemplateDecl(ClassTemplateDecl *D);
    virtual void VisitClassTemplateSpecializationDecl(
                                            ClassTemplateSpecializationDecl *D);
    virtual void VisitClassTemplatePartialSpecializationDecl(
                                     ClassTemplatePartialSpecializationDecl *D);
    virtual void VisitObjCMethodDecl(ObjCMethodDecl *D);
    virtual void VisitObjCImplementationDecl(ObjCImplementationDecl *D);
    virtual void VisitObjCInterfaceDecl(ObjCInterfaceDecl *D);
    virtual void VisitObjCProtocolDecl(ObjCProtocolDecl *D);
    virtual void VisitObjCCategoryImplDecl(ObjCCategoryImplDecl *D);
    virtual void VisitObjCCategoryDecl(ObjCCategoryDecl *D);
    virtual void VisitObjCCompatibleAliasDecl(ObjCCompatibleAliasDecl *D);
    virtual void VisitObjCPropertyDecl(ObjCPropertyDecl *D);
    virtual void VisitObjCPropertyImplDecl(ObjCPropertyImplDecl *D);
    virtual void VisitUnresolvedUsingTypenameDecl(UnresolvedUsingTypenameDecl *D);
    virtual void VisitUnresolvedUsingValueDecl(UnresolvedUsingValueDecl *D);
    virtual void VisitUsingDecl(UsingDecl *D);
    virtual void VisitUsingEnumDecl(UsingEnumDecl *D);
    virtual void VisitUsingShadowDecl(UsingShadowDecl *D);
    virtual void VisitOMPThreadPrivateDecl(OMPThreadPrivateDecl *D);
    virtual void VisitOMPAllocateDecl(OMPAllocateDecl *D);
    virtual void VisitOMPRequiresDecl(OMPRequiresDecl *D);
    virtual void VisitOMPDeclareReductionDecl(OMPDeclareReductionDecl *D);
    virtual void VisitOMPDeclareMapperDecl(OMPDeclareMapperDecl *D);
    virtual void VisitOMPCapturedExprDecl(OMPCapturedExprDecl *D);
    virtual void VisitTemplateTypeParmDecl(const TemplateTypeParmDecl *TTP);
    virtual void VisitNonTypeTemplateParmDecl(const NonTypeTemplateParmDecl *NTTP);
    virtual void VisitHLSLBufferDecl(HLSLBufferDecl *D);

    virtual void printTemplateParameters(const TemplateParameterList *Params,
                                 bool OmitTemplateKW = false);
    virtual void printTemplateArguments(llvm::ArrayRef<TemplateArgument> Args,
                                const TemplateParameterList *Params);
    virtual void printTemplateArguments(llvm::ArrayRef<TemplateArgumentLoc> Args,
                                const TemplateParameterList *Params);
    enum class AttrPosAsWritten { Default = 0, Left, Right };
    virtual bool
    prettyPrintAttributes(const Decl *D,
                          AttrPosAsWritten Pos = AttrPosAsWritten::Default);
    virtual void prettyPrintPragmas(Decl *D);
    virtual void printDeclType(QualType T, StringRef DeclName, bool Pack = false);

    bool isImplicitThis(const Expr *E);
    void printPretty(const Stmt* S, raw_ostream &Out, PrinterHelper *Helper,
                           const PrintingPolicy &Policy, unsigned Indentation,
                           StringRef NL, const ASTContext *Context) ;
    void printPrettyControlled(Stmt* S, raw_ostream &Out, PrinterHelper *Helper,
                                     const PrintingPolicy &Policy,
                                     unsigned Indentation, StringRef NL,
                                     const ASTContext *Context) ;
    void print(Decl *D, raw_ostream &Out, const PrintingPolicy &Policy,
                     unsigned Indentation, bool PrintInstantiation) ;
    QualType GetBaseType(QualType T) ;
    QualType getDeclType(Decl* D) ;
    void printGroup(Decl** Begin, unsigned NumDecls,
                          raw_ostream &Out, const PrintingPolicy &Policy,
                          unsigned Indentation) ;
    void printExplicitSpecifier(ExplicitSpecifier ES, llvm::raw_ostream &Out,
                                       PrintingPolicy &Policy, unsigned Indentation,
                                       const ASTContext &Context) ;
  };

void DeclPrinter::printPretty(const Stmt* S, raw_ostream &Out, PrinterHelper *Helper,
                       const PrintingPolicy &Policy, unsigned Indentation,
                       StringRef NL, const ASTContext *Context) {
  this->Visit(const_cast<Stmt *>(S));
}

void DeclPrinter::printPrettyControlled(Stmt* S, raw_ostream &Out, PrinterHelper *Helper,
                                 const PrintingPolicy &Policy,
                                 unsigned Indentation, StringRef NL,
                                 const ASTContext *Context) {
  this->PrintControlledStmt(const_cast<Stmt *>(S));
}

void DeclPrinter::print(Decl *D, raw_ostream &Out, const PrintingPolicy &Policy,
                 unsigned Indentation, bool PrintInstantiation) {
  this->Base::Visit(const_cast<Decl*>(D));
}


QualType DeclPrinter::GetBaseType(QualType T) {
  // FIXME: This should be on the Type class!
  QualType BaseType = T;
  while (!BaseType->isSpecifierType()) {
    if (const PointerType *PTy = BaseType->getAs<PointerType>())
      BaseType = PTy->getPointeeType();
    else if (const ObjCObjectPointerType *OPT =
                 BaseType->getAs<ObjCObjectPointerType>())
      BaseType = OPT->getPointeeType();
    else if (const BlockPointerType *BPy = BaseType->getAs<BlockPointerType>())
      BaseType = BPy->getPointeeType();
    else if (const ArrayType *ATy = dyn_cast<ArrayType>(BaseType))
      BaseType = ATy->getElementType();
    else if (const FunctionType *FTy = BaseType->getAs<FunctionType>())
      BaseType = FTy->getReturnType();
    else if (const VectorType *VTy = BaseType->getAs<VectorType>())
      BaseType = VTy->getElementType();
    else if (const ReferenceType *RTy = BaseType->getAs<ReferenceType>())
      BaseType = RTy->getPointeeType();
    else if (const AutoType *ATy = BaseType->getAs<AutoType>())
      BaseType = ATy->getDeducedType();
    else if (const ParenType *PTy = BaseType->getAs<ParenType>())
      BaseType = PTy->desugar();
    else
      // This must be a syntax error.
      break;
  }
  return BaseType;
}

QualType DeclPrinter::getDeclType(Decl* D) {
  if (TypedefNameDecl* TDD = dyn_cast<TypedefNameDecl>(D))
    return TDD->getUnderlyingType();
  if (ValueDecl* VD = dyn_cast<ValueDecl>(D))
    return VD->getType();
  return QualType();
}

void DeclPrinter::printGroup(Decl** Begin, unsigned NumDecls,
                      raw_ostream &Out, const PrintingPolicy &Policy,
                      unsigned Indentation) {
  if (NumDecls == 1) {
    print(*Begin, Out, Policy, Indentation, false);
    return;
  }

  Decl** End = Begin + NumDecls;
  TagDecl* TD = dyn_cast<TagDecl>(*Begin);
  if (TD)
    ++Begin;

  PrintingPolicy SubPolicy(Policy);

  bool isFirst = true;
  for ( ; Begin != End; ++Begin) {
    if (isFirst) {
      if(TD)
        SubPolicy.IncludeTagDefinition = true;
      SubPolicy.SuppressSpecifiers = false;
      isFirst = false;
    } else {
      if (!isFirst) Out << ", ";
      SubPolicy.IncludeTagDefinition = false;
      SubPolicy.SuppressSpecifiers = true;
    }

    print(*Begin, Out, SubPolicy, Indentation, false);
  }
}

raw_ostream& DeclPrinter::Indent(unsigned Indentation) {
  for (unsigned i = 0; i != Indentation; ++i)
    Out << "  ";
  return Out;
}

static DeclPrinter::AttrPosAsWritten getPosAsWritten(const Attr *A,
                                                     const Decl *D) {
  SourceLocation ALoc = A->getLoc();
  SourceLocation DLoc = D->getLocation();
  const ASTContext &C = D->getASTContext();
  if (ALoc.isInvalid() || DLoc.isInvalid())
    return DeclPrinter::AttrPosAsWritten::Left;

  if (C.getSourceManager().isBeforeInTranslationUnit(ALoc, DLoc))
    return DeclPrinter::AttrPosAsWritten::Left;

  return DeclPrinter::AttrPosAsWritten::Right;
}

// returns true if an attribute was printed.
bool DeclPrinter::prettyPrintAttributes(const Decl *D,
                                        AttrPosAsWritten Pos /*=Default*/) {
  bool hasPrinted = false;

  if (D->hasAttrs()) {
    const AttrVec &Attrs = D->getAttrs();
    for (auto *A : Attrs) {
      if (A->isInherited() || A->isImplicit())
        continue;
      // Print out the keyword attributes, they aren't regular attributes.
      if (Policy.PolishForDeclaration && !A->isKeywordAttribute())
        continue;
      switch (A->getKind()) {
#define ATTR(X)
#define PRAGMA_SPELLING_ATTR(X) case attr::X:
#include "clang/Basic/AttrList.inc"
        break;
      default:
        AttrPosAsWritten APos = getPosAsWritten(A, D);
        assert(APos != AttrPosAsWritten::Default &&
               "Default not a valid for an attribute location");
        if (Pos == AttrPosAsWritten::Default || Pos == APos) {
          if (Pos != AttrPosAsWritten::Left)
            Out << ' ';
          A->printPretty(Out, Policy);
          hasPrinted = true;
          if (Pos == AttrPosAsWritten::Left)
            Out << ' ';
        }
        break;
      }
    }
  }
  return hasPrinted;
}

void DeclPrinter::prettyPrintPragmas(Decl *D) {
  if (Policy.PolishForDeclaration)
    return;

  if (D->hasAttrs()) {
    AttrVec &Attrs = D->getAttrs();
    for (auto *A : Attrs) {
      switch (A->getKind()) {
#define ATTR(X)
#define PRAGMA_SPELLING_ATTR(X) case attr::X:
#include "clang/Basic/AttrList.inc"
        A->printPretty(Out, Policy);
        Indent();
        break;
      default:
        break;
      }
    }
  }
}

void DeclPrinter::printDeclType(QualType T, StringRef DeclName, bool Pack) {
  // Normally, a PackExpansionType is written as T[3]... (for instance, as a
  // template argument), but if it is the type of a declaration, the ellipsis
  // is placed before the name being declared.
  if (auto *PET = T->getAs<PackExpansionType>()) {
    Pack = true;
    T = PET->getPattern();
  }
  T.print(Out, Policy, (Pack ? "..." : "") + DeclName, Indentation);
}

void DeclPrinter::ProcessDeclGroup(SmallVectorImpl<Decl*>& Decls) {
  this->Indent();
  printGroup(Decls.data(), Decls.size(), Out, Policy, Indentation);
  Out << ";\n";
  Decls.clear();

}

void DeclPrinter::Print(AccessSpecifier AS) {
  const auto AccessSpelling = getAccessSpelling(AS);
  if (AccessSpelling.empty())
    llvm_unreachable("No access specifier!");
  Out << AccessSpelling;
}

void DeclPrinter::PrintConstructorInitializers(CXXConstructorDecl *CDecl,
                                               std::string &Proto) {
  bool HasInitializerList = false;
  for (const auto *BMInitializer : CDecl->inits()) {
    if (BMInitializer->isInClassMemberInitializer())
      continue;
    if (!BMInitializer->isWritten())
      continue;

    if (!HasInitializerList) {
      Proto += " : ";
      Out << Proto;
      Proto.clear();
      HasInitializerList = true;
    } else
      Out << ", ";

    if (BMInitializer->isAnyMemberInitializer()) {
      FieldDecl *FD = BMInitializer->getAnyMember();
      Out << *FD;
    } else if (BMInitializer->isDelegatingInitializer()) {
      Out << CDecl->getNameAsString();
    } else {
      Out << QualType(BMInitializer->getBaseClass(), 0).getAsString(Policy);
    }

    if (Expr *Init = BMInitializer->getInit()) {
      bool OutParens = !isa<InitListExpr>(Init);

      if (OutParens)
        Out << "(";

      if (ExprWithCleanups *Tmp = dyn_cast<ExprWithCleanups>(Init))
        Init = Tmp->getSubExpr();

      Init = Init->IgnoreParens();

      Expr *SimpleInit = nullptr;
      Expr **Args = nullptr;
      unsigned NumArgs = 0;
      if (ParenListExpr *ParenList = dyn_cast<ParenListExpr>(Init)) {
        Args = ParenList->getExprs();
        NumArgs = ParenList->getNumExprs();
      } else if (CXXConstructExpr *Construct =
                     dyn_cast<CXXConstructExpr>(Init)) {
        Args = Construct->getArgs();
        NumArgs = Construct->getNumArgs();
      } else
        SimpleInit = Init;

      if (SimpleInit)
        printPretty(SimpleInit, Out, nullptr, Policy, Indentation, "\n",
                                &Context);
      else {
        for (unsigned I = 0; I != NumArgs; ++I) {
          assert(Args[I] != nullptr && "Expected non-null Expr");
          if (isa<CXXDefaultArgExpr>(Args[I]))
            break;

          if (I)
            Out << ", ";
          printPretty(Args[I], Out, nullptr, Policy, Indentation, "\n",
                               &Context);
        }
      }

      if (OutParens)
        Out << ")";
    } else {
      Out << "()";
    }

    if (BMInitializer->isPackExpansion())
      Out << "...";
  }
}

//----------------------------------------------------------------------------
// Common C declarations
//----------------------------------------------------------------------------

void DeclPrinter::VisitDeclContext(DeclContext *DC, bool Indent) {
  if (Policy.TerseOutput)
    return;

  if (Indent)
    Indentation += Policy.Indentation;

  SmallVector<Decl*, 2> Decls;
  for (DeclContext::decl_iterator D = DC->decls_begin(), DEnd = DC->decls_end();
       D != DEnd; ++D) {

    // Don't print ObjCIvarDecls, as they are printed when visiting the
    // containing ObjCInterfaceDecl.
    if (isa<ObjCIvarDecl>(*D))
      continue;

    // Skip over implicit declarations in pretty-printing mode.
    if (D->isImplicit())
      continue;

    // Don't print implicit specializations, as they are printed when visiting
    // corresponding templates.
    if (auto *FD = dyn_cast<FunctionDecl>(*D))
      if (FD->getTemplateSpecializationKind() == TSK_ImplicitInstantiation &&
          !isa<ClassTemplateSpecializationDecl>(DC))
        continue;

    // The next bits of code handle stuff like "struct {int x;} a,b"; we're
    // forced to merge the declarations because there's no other way to
    // refer to the struct in question.  When that struct is named instead, we
    // also need to merge to avoid splitting off a stand-alone struct
    // declaration that produces the warning ext_no_declarators in some
    // contexts.
    //
    // This limited merging is safe without a bunch of other checks because it
    // only merges declarations directly referring to the tag, not typedefs.
    //
    // Check whether the current declaration should be grouped with a previous
    // non-free-standing tag declaration.
    QualType CurDeclType = getDeclType(*D);
    if (!Decls.empty() && !CurDeclType.isNull()) {
      QualType BaseType = GetBaseType(CurDeclType);
      if (!BaseType.isNull() && isa<ElaboratedType>(BaseType) &&
          cast<ElaboratedType>(BaseType)->getOwnedTagDecl() == Decls[0]) {
        Decls.push_back(*D);
        continue;
      }
    }

    // If we have a merged group waiting to be handled, handle it now.
    if (!Decls.empty())
      ProcessDeclGroup(Decls);

    // If the current declaration is not a free standing declaration, save it
    // so we can merge it with the subsequent declaration(s) using it.
    if (isa<TagDecl>(*D) && !cast<TagDecl>(*D)->isFreeStanding()) {
      Decls.push_back(*D);
      continue;
    }

    if (isa<AccessSpecDecl>(*D)) {
      Indentation -= Policy.Indentation;
      this->Indent();
      Print(D->getAccess());
      Out << ":\n";
      Indentation += Policy.Indentation;
      continue;
    }

    this->Indent();
    Base::Visit(*D);

    // FIXME: Need to be able to tell the DeclPrinter when
    const char *Terminator = nullptr;
    if (isa<OMPThreadPrivateDecl>(*D) || isa<OMPDeclareReductionDecl>(*D) ||
        isa<OMPDeclareMapperDecl>(*D) || isa<OMPRequiresDecl>(*D) ||
        isa<OMPAllocateDecl>(*D))
      Terminator = nullptr;
    else if (isa<ObjCMethodDecl>(*D) && cast<ObjCMethodDecl>(*D)->hasBody())
      Terminator = nullptr;
    else if (auto *FD = dyn_cast<FunctionDecl>(*D)) {
      if (FD->doesThisDeclarationHaveABody() && !FD->isDefaulted())
        Terminator = nullptr;
      else
        Terminator = ";";
    } else if (auto *TD = dyn_cast<FunctionTemplateDecl>(*D)) {
      if (TD->getTemplatedDecl()->doesThisDeclarationHaveABody())
        Terminator = nullptr;
      else
        Terminator = ";";
    } else if (isa<NamespaceDecl, LinkageSpecDecl, ObjCImplementationDecl,
                   ObjCInterfaceDecl, ObjCProtocolDecl, ObjCCategoryImplDecl,
                   ObjCCategoryDecl, HLSLBufferDecl>(*D))
      Terminator = nullptr;
    else if (isa<EnumConstantDecl>(*D)) {
      DeclContext::decl_iterator Next = D;
      ++Next;
      if (Next != DEnd)
        Terminator = ",";
    } else
      Terminator = ";";

    if (Terminator)
      Out << Terminator;
    if (!Policy.TerseOutput &&
        ((isa<FunctionDecl>(*D) &&
          cast<FunctionDecl>(*D)->doesThisDeclarationHaveABody()) ||
         (isa<FunctionTemplateDecl>(*D) &&
          cast<FunctionTemplateDecl>(*D)->getTemplatedDecl()->doesThisDeclarationHaveABody())))
      ; // StmtPrinter already added '\n' after CompoundStmt.
    else
      Out << "\n";

    // Declare target attribute is special one, natural spelling for the pragma
    // assumes "ending" construct so print it here.
    if (D->hasAttr<OMPDeclareTargetDeclAttr>())
      Out << "#pragma omp end declare target\n";
  }

  if (!Decls.empty())
    ProcessDeclGroup(Decls);

  if (Indent)
    Indentation -= Policy.Indentation;
}

void DeclPrinter::VisitTranslationUnitDecl(TranslationUnitDecl *D) {
  VisitDeclContext(D, false);
}

void DeclPrinter::VisitTypedefDecl(TypedefDecl *D) {
  if (!Policy.SuppressSpecifiers) {
    Out << "typedef ";

    if (D->isModulePrivate())
      Out << "__module_private__ ";
  }
  QualType Ty = D->getTypeSourceInfo()->getType();
  Ty.print(Out, Policy, D->getName(), Indentation);
  prettyPrintAttributes(D);
}

void DeclPrinter::VisitTypeAliasDecl(TypeAliasDecl *D) {
  Out << "using " << *D;
  prettyPrintAttributes(D);
  Out << " = " << D->getTypeSourceInfo()->getType().getAsString(Policy);
}

void DeclPrinter::VisitEnumDecl(EnumDecl *D) {
  if (!Policy.SuppressSpecifiers && D->isModulePrivate())
    Out << "__module_private__ ";
  Out << "enum";
  if (D->isScoped()) {
    if (D->isScopedUsingClassTag())
      Out << " class";
    else
      Out << " struct";
  }

  prettyPrintAttributes(D);

  if (D->getDeclName())
    Out << ' ' << D->getDeclName();

  if (D->isFixed())
    Out << " : " << D->getIntegerType().stream(Policy);

  if (D->isCompleteDefinition()) {
    Out << " {\n";
    VisitDeclContext(D);
    Indent() << "}";
  }
}

void DeclPrinter::VisitRecordDecl(RecordDecl *D) {
  if (!Policy.SuppressSpecifiers && D->isModulePrivate())
    Out << "__module_private__ ";
  Out << D->getKindName();

  prettyPrintAttributes(D);

  if (D->getIdentifier())
    Out << ' ' << *D;

  if (D->isCompleteDefinition()) {
    Out << " {\n";
    VisitDeclContext(D);
    Indent() << "}";
  }
}

void DeclPrinter::VisitEnumConstantDecl(EnumConstantDecl *D) {
  Out << *D;
  prettyPrintAttributes(D);
  if (Expr *Init = D->getInitExpr()) {
    Out << " = ";
    printPretty(Init, Out, nullptr, Policy, Indentation, "\n", &Context);
  }
}

void DeclPrinter::printExplicitSpecifier(ExplicitSpecifier ES, llvm::raw_ostream &Out,
                                   PrintingPolicy &Policy, unsigned Indentation,
                                   const ASTContext &Context) {
  std::string Proto = "explicit";
  llvm::raw_string_ostream EOut(Proto);
  if (ES.getExpr()) {
    EOut << "(";
    printPretty(ES.getExpr(), EOut, nullptr, Policy, Indentation, "\n",
                              &Context);
    EOut << ")";
  }
  EOut << " ";
  EOut.flush();
  Out << Proto;
}

static void MaybePrintTagKeywordIfSupressingScopes(PrintingPolicy &Policy,
                                                   QualType T,
                                                   llvm::raw_ostream &Out) {
  StringRef prefix = T->isClassType()       ? "class "
                     : T->isStructureType() ? "struct "
                     : T->isUnionType()     ? "union "
                                            : "";
  Out << prefix;
}

void DeclPrinter::VisitFunctionDecl(FunctionDecl *D) {
  if (!D->getDescribedFunctionTemplate() &&
      !D->isFunctionTemplateSpecialization()) {
    prettyPrintPragmas(D);
    prettyPrintAttributes(D, AttrPosAsWritten::Left);
  }

  if (D->isFunctionTemplateSpecialization())
    Out << "template<> ";
  else if (!D->getDescribedFunctionTemplate()) {
    for (unsigned I = 0, NumTemplateParams = D->getNumTemplateParameterLists();
         I < NumTemplateParams; ++I)
      printTemplateParameters(D->getTemplateParameterList(I));
  }

  CXXConstructorDecl *CDecl = dyn_cast<CXXConstructorDecl>(D);
  CXXConversionDecl *ConversionDecl = dyn_cast<CXXConversionDecl>(D);
  CXXDeductionGuideDecl *GuideDecl = dyn_cast<CXXDeductionGuideDecl>(D);
  if (!Policy.SuppressSpecifiers) {
    switch (D->getStorageClass()) {
    case SC_None: break;
    case SC_Extern: Out << "extern "; break;
    case SC_Static: Out << "static "; break;
    case SC_PrivateExtern: Out << "__private_extern__ "; break;
    case SC_Auto: case SC_Register:
      llvm_unreachable("invalid for functions");
    }

    if (D->isInlineSpecified())  Out << "inline ";
    if (D->isVirtualAsWritten()) Out << "virtual ";
    if (D->isModulePrivate())    Out << "__module_private__ ";
    if (D->isConstexprSpecified() && !D->isExplicitlyDefaulted())
      Out << "constexpr ";
    if (D->isConsteval())        Out << "consteval ";
    else if (D->isImmediateFunction())
      Out << "immediate ";
    ExplicitSpecifier ExplicitSpec = ExplicitSpecifier::getFromDecl(D);
    if (ExplicitSpec.isSpecified())
      printExplicitSpecifier(ExplicitSpec, Out, Policy, Indentation, Context);
  }

  PrintingPolicy SubPolicy(Policy);
  SubPolicy.SuppressSpecifiers = false;
  std::string Proto;

  if (Policy.FullyQualifiedName) {
    Proto += D->getQualifiedNameAsString();
  } else {
    llvm::raw_string_ostream OS(Proto);
    if (!Policy.SuppressScope) {
      if (const NestedNameSpecifier *NS = D->getQualifier()) {
        NS->print(OS, Policy);
      }
    }
    D->getNameInfo().printName(OS, Policy);
  }

  if (GuideDecl)
    Proto = GuideDecl->getDeducedTemplate()->getDeclName().getAsString();
  if (D->isFunctionTemplateSpecialization()) {
    llvm::raw_string_ostream POut(Proto);
    const auto *TArgAsWritten = D->getTemplateSpecializationArgsAsWritten();
    if (TArgAsWritten && !Policy.PrintCanonicalTypes)
      printTemplateArguments(TArgAsWritten->arguments(), nullptr);
    else if (const TemplateArgumentList *TArgs =
                 D->getTemplateSpecializationArgs())
      printTemplateArguments(TArgs->asArray(), nullptr);
  }

  QualType Ty = D->getType();
  while (const ParenType *PT = dyn_cast<ParenType>(Ty)) {
    Proto = '(' + Proto + ')';
    Ty = PT->getInnerType();
  }

  if (const FunctionType *AFT = Ty->getAs<FunctionType>()) {
    const FunctionProtoType *FT = nullptr;
    if (D->hasWrittenPrototype())
      FT = dyn_cast<FunctionProtoType>(AFT);

    Proto += "(";
    if (FT) {
      llvm::raw_string_ostream POut(Proto);
      for (unsigned i = 0, e = D->getNumParams(); i != e; ++i) {
        if (i) POut << ", ";
        VisitParmVarDecl(D->getParamDecl(i));
      }

      if (FT->isVariadic()) {
        if (D->getNumParams()) POut << ", ";
        POut << "...";
      } else if (!D->getNumParams() && !Context.getLangOpts().CPlusPlus) {
        // The function has a prototype, so it needs to retain the prototype
        // in C.
        POut << "void";
      }
    } else if (D->doesThisDeclarationHaveABody() && !D->hasPrototype()) {
      for (unsigned i = 0, e = D->getNumParams(); i != e; ++i) {
        if (i)
          Proto += ", ";
        Proto += D->getParamDecl(i)->getNameAsString();
      }
    }

    Proto += ")";

    if (FT) {
      if (FT->isConst())
        Proto += " const";
      if (FT->isVolatile())
        Proto += " volatile";
      if (FT->isRestrict())
        Proto += " restrict";

      switch (FT->getRefQualifier()) {
      case RQ_None:
        break;
      case RQ_LValue:
        Proto += " &";
        break;
      case RQ_RValue:
        Proto += " &&";
        break;
      }
    }

    if (FT && FT->hasDynamicExceptionSpec()) {
      Proto += " throw(";
      if (FT->getExceptionSpecType() == EST_MSAny)
        Proto += "...";
      else
        for (unsigned I = 0, N = FT->getNumExceptions(); I != N; ++I) {
          if (I)
            Proto += ", ";

          Proto += FT->getExceptionType(I).getAsString(SubPolicy);
        }
      Proto += ")";
    } else if (FT && isNoexceptExceptionSpec(FT->getExceptionSpecType())) {
      Proto += " noexcept";
      if (isComputedNoexcept(FT->getExceptionSpecType())) {
        Proto += "(";
        llvm::raw_string_ostream EOut(Proto);
        printPretty(FT->getNoexceptExpr(), EOut, nullptr, SubPolicy,
                                           Indentation, "\n", &Context);
        EOut.flush();
        Proto += ")";
      }
    }

    if (CDecl) {
      if (!Policy.TerseOutput)
        PrintConstructorInitializers(CDecl, Proto);
    } else if (!ConversionDecl && !isa<CXXDestructorDecl>(D)) {
      if (FT && FT->hasTrailingReturn()) {
        if (!GuideDecl)
          Out << "auto ";
        Out << Proto << " -> ";
        Proto.clear();
      }
      if (!Policy.SuppressTagKeyword && Policy.SuppressScope &&
          !Policy.SuppressUnwrittenScope)
        MaybePrintTagKeywordIfSupressingScopes(Policy, AFT->getReturnType(),
                                               Out);
      AFT->getReturnType().print(Out, Policy, Proto);
      Proto.clear();
    }
    Out << Proto;

    if (Expr *TrailingRequiresClause = D->getTrailingRequiresClause()) {
      Out << " requires ";
      printPretty(TrailingRequiresClause, Out, nullptr, SubPolicy, Indentation,
                                          "\n", &Context);
    }
  } else {
    Ty.print(Out, Policy, Proto);
  }

  prettyPrintAttributes(D, AttrPosAsWritten::Right);

  if (D->isPureVirtual())
    Out << " = 0";
  else if (D->isDeletedAsWritten()) {
    Out << " = delete";
    if (const StringLiteral *M = D->getDeletedMessage()) {
      Out << "(";
      M->outputString(Out);
      Out << ")";
    }
  } else if (D->isExplicitlyDefaulted())
    Out << " = default";
  else if (D->doesThisDeclarationHaveABody()) {
    if (!Policy.TerseOutput) {
      if (!D->hasPrototype() && D->getNumParams()) {
        // This is a K&R function definition, so we need to print the
        // parameters.
        Out << '\n';
        Indentation += Policy.Indentation;
        for (unsigned i = 0, e = D->getNumParams(); i != e; ++i) {
          Indent();
          VisitParmVarDecl(D->getParamDecl(i));
          Out << ";\n";
        }
        Indentation -= Policy.Indentation;
      }

      if (D->getBody())
        printPrettyControlled(D->getBody(), Out, nullptr, SubPolicy, Indentation, "\n",
                                  &Context);
    } else {
      if (!Policy.TerseOutput && isa<CXXConstructorDecl>(*D))
        Out << " {}";
    }
  }
}

void DeclPrinter::VisitFriendDecl(FriendDecl *D) {
  if (TypeSourceInfo *TSI = D->getFriendType()) {
    unsigned NumTPLists = D->getFriendTypeNumTemplateParameterLists();
    for (unsigned i = 0; i < NumTPLists; ++i)
      printTemplateParameters(D->getFriendTypeTemplateParameterList(i));
    Out << "friend ";
    Out << " " << TSI->getType().getAsString(Policy);
  }
  else if (FunctionDecl *FD =
      dyn_cast<FunctionDecl>(D->getFriendDecl())) {
    Out << "friend ";
    VisitFunctionDecl(FD);
  }
  else if (FunctionTemplateDecl *FTD =
           dyn_cast<FunctionTemplateDecl>(D->getFriendDecl())) {
    Out << "friend ";
    VisitFunctionTemplateDecl(FTD);
  }
  else if (ClassTemplateDecl *CTD =
           dyn_cast<ClassTemplateDecl>(D->getFriendDecl())) {
    Out << "friend ";
    VisitRedeclarableTemplateDecl(CTD);
  }
}

void DeclPrinter::VisitFieldDecl(FieldDecl *D) {
  // FIXME: add printing of pragma attributes if required.
  if (!Policy.SuppressSpecifiers && D->isMutable())
    Out << "mutable ";
  if (!Policy.SuppressSpecifiers && D->isModulePrivate())
    Out << "__module_private__ ";

  Out << D->getASTContext().getUnqualifiedObjCPointerType(D->getType()).
         stream(Policy, D->getName(), Indentation);

  if (D->isBitField()) {
    Out << " : ";
    printPretty(D->getBitWidth(), Out, nullptr, Policy, Indentation, "\n",
                                  &Context);
  }

  Expr *Init = D->getInClassInitializer();
  if (!Policy.SuppressInitializers && Init) {
    if (D->getInClassInitStyle() == ICIS_ListInit)
      Out << " ";
    else
      Out << " = ";
    printPretty(Init, Out, nullptr, Policy, Indentation, "\n", &Context);
  }
  prettyPrintAttributes(D);
}

void DeclPrinter::VisitLabelDecl(LabelDecl *D) {
  Out << *D << ":";
}

void DeclPrinter::VisitVarDecl(VarDecl *D) {
  prettyPrintPragmas(D);

  prettyPrintAttributes(D, AttrPosAsWritten::Left);

  if (const auto *Param = dyn_cast<ParmVarDecl>(D);
      Param && Param->isExplicitObjectParameter())
    Out << "this ";

  QualType T = D->getTypeSourceInfo()
    ? D->getTypeSourceInfo()->getType()
    : D->getASTContext().getUnqualifiedObjCPointerType(D->getType());

  if (!Policy.SuppressSpecifiers) {
    StorageClass SC = D->getStorageClass();
    if (SC != SC_None)
      Out << VarDecl::getStorageClassSpecifierString(SC) << " ";

    switch (D->getTSCSpec()) {
    case TSCS_unspecified:
      break;
    case TSCS___thread:
      Out << "__thread ";
      break;
    case TSCS__Thread_local:
      Out << "_Thread_local ";
      break;
    case TSCS_thread_local:
      Out << "thread_local ";
      break;
    }

    if (D->isModulePrivate())
      Out << "__module_private__ ";

    if (D->isConstexpr()) {
      Out << "constexpr ";
      T.removeLocalConst();
    }
  }

  if (!Policy.SuppressTagKeyword && Policy.SuppressScope &&
      !Policy.SuppressUnwrittenScope)
    MaybePrintTagKeywordIfSupressingScopes(Policy, T, Out);

  printDeclType(T, (isa<ParmVarDecl>(D) && Policy.CleanUglifiedParameters &&
                    D->getIdentifier())
                       ? D->getIdentifier()->deuglifiedName()
                       : D->getName());

  prettyPrintAttributes(D, AttrPosAsWritten::Right);

  Expr *Init = D->getInit();
  if (!Policy.SuppressInitializers && Init) {
    bool ImplicitInit = false;
    if (D->isCXXForRangeDecl()) {
      // FIXME: We should print the range expression instead.
      ImplicitInit = true;
    } else if (CXXConstructExpr *Construct =
                   dyn_cast<CXXConstructExpr>(Init->IgnoreImplicit())) {
      if (D->getInitStyle() == VarDecl::CallInit &&
          !Construct->isListInitialization()) {
        ImplicitInit = Construct->getNumArgs() == 0 ||
                       Construct->getArg(0)->isDefaultArgument();
      }
    }
    if (!ImplicitInit) {
      if ((D->getInitStyle() == VarDecl::CallInit) && !isa<ParenListExpr>(Init))
        Out << "(";
      else if (D->getInitStyle() == VarDecl::CInit) {
        Out << " = ";
      }
      PrintingPolicy SubPolicy(Policy);
      SubPolicy.SuppressSpecifiers = false;
      SubPolicy.IncludeTagDefinition = false;
      printPretty(Init, Out, nullptr, SubPolicy, Indentation, "\n", &Context);
      if ((D->getInitStyle() == VarDecl::CallInit) && !isa<ParenListExpr>(Init))
        Out << ")";
    }
  }
}

void DeclPrinter::VisitParmVarDecl(ParmVarDecl *D) {
  VisitVarDecl(D);
}

void DeclPrinter::VisitFileScopeAsmDecl(FileScopeAsmDecl *D) {
  Out << "__asm (";
  printPretty(D->getAsmString(), Out, nullptr, Policy, Indentation, "\n",
                                 &Context);
  Out << ")";
}

void DeclPrinter::VisitTopLevelStmtDecl(TopLevelStmtDecl *D) {
  assert(D->getStmt());
  printPretty(D->getStmt(), Out, nullptr, Policy, Indentation, "\n", &Context);
}

void DeclPrinter::VisitImportDecl(ImportDecl *D) {
  Out << "@import " << D->getImportedModule()->getFullModuleName()
      << ";\n";
}

void DeclPrinter::VisitStaticAssertDecl(StaticAssertDecl *D) {
  Out << "static_assert(";
  printPretty(D->getAssertExpr(), Out, nullptr, Policy, Indentation, "\n",
                                  &Context);
  if (Expr *E = D->getMessage()) {
    Out << ", ";
    printPretty(E, Out, nullptr, Policy, Indentation, "\n", &Context);
  }
  Out << ")";
}

//----------------------------------------------------------------------------
// C++ declarations
//----------------------------------------------------------------------------
void DeclPrinter::VisitNamespaceDecl(NamespaceDecl *D) {
  if (D->isInline())
    Out << "inline ";

  Out << "namespace ";
  if (D->getDeclName())
    Out << D->getDeclName() << ' ';
  Out << "{\n";

  VisitDeclContext(D);
  Indent() << "}";
}

void DeclPrinter::VisitUsingDirectiveDecl(UsingDirectiveDecl *D) {
  Out << "using namespace ";
  if (D->getQualifier())
    D->getQualifier()->print(Out, Policy);
  Out << *D->getNominatedNamespaceAsWritten();
}

void DeclPrinter::VisitNamespaceAliasDecl(NamespaceAliasDecl *D) {
  Out << "namespace " << *D << " = ";
  if (D->getQualifier())
    D->getQualifier()->print(Out, Policy);
  Out << *D->getAliasedNamespace();
}

void DeclPrinter::VisitEmptyDecl(EmptyDecl *D) {
  prettyPrintAttributes(D);
}

void DeclPrinter::VisitCXXRecordDecl(CXXRecordDecl *D) {
  // FIXME: add printing of pragma attributes if required.
  if (!Policy.SuppressSpecifiers && D->isModulePrivate())
    Out << "__module_private__ ";

  Out << D->getKindName() << ' ';

  // FIXME: Move before printing the decl kind to match the behavior of the
  // attribute printing for variables and function where they are printed first.
  if (prettyPrintAttributes(D, AttrPosAsWritten::Left))
    Out << ' ';

  if (D->getIdentifier()) {
    if (auto *NNS = D->getQualifier())
      NNS->print(Out, Policy);
    Out << *D;

    if (auto *S = dyn_cast<ClassTemplateSpecializationDecl>(D)) {
      const TemplateParameterList *TParams =
          S->getSpecializedTemplate()->getTemplateParameters();
      const ASTTemplateArgumentListInfo *TArgAsWritten =
          S->getTemplateArgsAsWritten();
      if (TArgAsWritten && !Policy.PrintCanonicalTypes)
        printTemplateArguments(TArgAsWritten->arguments(), TParams);
      else
        printTemplateArguments(S->getTemplateArgs().asArray(), TParams);
    }
  }

  prettyPrintAttributes(D, AttrPosAsWritten::Right);

  if (D->isCompleteDefinition()) {
    Out << ' ';
    // Print the base classes
    if (D->getNumBases()) {
      Out << ": ";
      for (CXXRecordDecl::base_class_iterator Base = D->bases_begin(),
             BaseEnd = D->bases_end(); Base != BaseEnd; ++Base) {
        if (Base != D->bases_begin())
          Out << ", ";

        if (Base->isVirtual())
          Out << "virtual ";

        AccessSpecifier AS = Base->getAccessSpecifierAsWritten();
        if (AS != AS_none) {
          Print(AS);
          Out << " ";
        }
        Out << Base->getType().getAsString(Policy);

        if (Base->isPackExpansion())
          Out << "...";
      }
      Out << ' ';
    }

    // Print the class definition
    // FIXME: Doesn't print access specifiers, e.g., "public:"
    if (Policy.TerseOutput) {
      Out << "{}";
    } else {
      Out << "{\n";
      VisitDeclContext(D);
      Indent() << "}";
    }
  }
}

void DeclPrinter::VisitLinkageSpecDecl(LinkageSpecDecl *D) {
  const char *l;
  if (D->getLanguage() == LinkageSpecLanguageIDs::C)
    l = "C";
  else {
    assert(D->getLanguage() == LinkageSpecLanguageIDs::CXX &&
           "unknown language in linkage specification");
    l = "C++";
  }

  Out << "extern \"" << l << "\" ";
  if (D->hasBraces()) {
    Out << "{\n";
    VisitDeclContext(D);
    Indent() << "}";
  } else
    Base::Visit(*D->decls_begin());
}

void DeclPrinter::printTemplateParameters(const TemplateParameterList *Params,
                                          bool OmitTemplateKW) {
  assert(Params);

  // Don't print invented template parameter lists.
  if (!Params->empty() && Params->getParam(0)->isImplicit())
    return;

  if (!OmitTemplateKW)
    Out << "template ";
  Out << '<';

  bool NeedComma = false;
  for (const Decl *Param : *Params) {
    if (Param->isImplicit())
      continue;

    if (NeedComma)
      Out << ", ";
    else
      NeedComma = true;

    if (const auto *TTP = dyn_cast<TemplateTypeParmDecl>(Param)) {
      VisitTemplateTypeParmDecl(TTP);
    } else if (const auto *NTTP = dyn_cast<NonTypeTemplateParmDecl>(Param)) {
      VisitNonTypeTemplateParmDecl(NTTP);
    } else if (const auto *TTPD = dyn_cast<TemplateTemplateParmDecl>(Param)) {
      VisitTemplateDecl(TTPD);
      // FIXME: print the default argument, if present.
    }
  }

  Out << '>';

  if (const Expr *RequiresClause = Params->getRequiresClause()) {
    Out << " requires ";
    printPretty(RequiresClause, Out, nullptr, Policy, Indentation, "\n",
                                &Context);
  }

  if (!OmitTemplateKW)
    Out << ' ';
}

void DeclPrinter::printTemplateArguments(ArrayRef<TemplateArgument> Args,
                                         const TemplateParameterList *Params) {
  Out << "<";
  for (size_t I = 0, E = Args.size(); I < E; ++I) {
    if (I)
      Out << ", ";
    if (!Params)
      Args[I].print(Policy, Out, /*IncludeType*/ true);
    else
      Args[I].print(Policy, Out,
                    TemplateParameterList::shouldIncludeTypeForArgument(
                        Policy, Params, I));
  }
  Out << ">";
}

void DeclPrinter::printTemplateArguments(ArrayRef<TemplateArgumentLoc> Args,
                                         const TemplateParameterList *Params) {
  Out << "<";
  for (size_t I = 0, E = Args.size(); I < E; ++I) {
    if (I)
      Out << ", ";
    if (!Params)
      Args[I].getArgument().print(Policy, Out, /*IncludeType*/ true);
    else
      Args[I].getArgument().print(
          Policy, Out,
          TemplateParameterList::shouldIncludeTypeForArgument(Policy, Params,
                                                              I));
  }
  Out << ">";
}

void DeclPrinter::VisitTemplateDecl(const TemplateDecl *D) {
  printTemplateParameters(D->getTemplateParameters());

  if (const TemplateTemplateParmDecl *TTP =
        dyn_cast<TemplateTemplateParmDecl>(D)) {
    if (TTP->wasDeclaredWithTypename())
      Out << "typename";
    else
      Out << "class";

    if (TTP->isParameterPack())
      Out << " ...";
    else if (TTP->getDeclName())
      Out << ' ';

    if (TTP->getDeclName()) {
      if (Policy.CleanUglifiedParameters && TTP->getIdentifier())
        Out << TTP->getIdentifier()->deuglifiedName();
      else
        Out << TTP->getDeclName();
    }
  } else if (auto *TD = D->getTemplatedDecl())
    Base::Visit(TD);
  else if (const auto *Concept = dyn_cast<ConceptDecl>(D)) {
    Out << "concept " << Concept->getName() << " = " ;
    printPretty(Concept->getConstraintExpr(), Out, nullptr, Policy, Indentation,
                                              "\n", &Context);
  }
}

void DeclPrinter::VisitFunctionTemplateDecl(FunctionTemplateDecl *D) {
  prettyPrintPragmas(D->getTemplatedDecl());
  // Print any leading template parameter lists.
  if (const FunctionDecl *FD = D->getTemplatedDecl()) {
    for (unsigned I = 0, NumTemplateParams = FD->getNumTemplateParameterLists();
         I < NumTemplateParams; ++I)
      printTemplateParameters(FD->getTemplateParameterList(I));
  }
  VisitRedeclarableTemplateDecl(D);
  // Declare target attribute is special one, natural spelling for the pragma
  // assumes "ending" construct so print it here.
  if (D->getTemplatedDecl()->hasAttr<OMPDeclareTargetDeclAttr>())
    Out << "#pragma omp end declare target\n";

  // Never print "instantiations" for deduction guides (they don't really
  // have them).
  if (PrintInstantiation &&
      !isa<CXXDeductionGuideDecl>(D->getTemplatedDecl())) {
    FunctionDecl *PrevDecl = D->getTemplatedDecl();
    const FunctionDecl *Def;
    if (PrevDecl->isDefined(Def) && Def != PrevDecl)
      return;
    for (auto *I : D->specializations())
      if (I->getTemplateSpecializationKind() == TSK_ImplicitInstantiation) {
        if (!PrevDecl->isThisDeclarationADefinition())
          Out << ";\n";
        Indent();
        prettyPrintPragmas(I);
        Base::Visit(I);
      }
  }
}

void DeclPrinter::VisitClassTemplateDecl(ClassTemplateDecl *D) {
  VisitRedeclarableTemplateDecl(D);

  if (PrintInstantiation) {
    for (auto *I : D->specializations())
      if (I->getSpecializationKind() == TSK_ImplicitInstantiation) {
        if (D->isThisDeclarationADefinition())
          Out << ";";
        Out << "\n";
        Indent();
        Base::Visit(I);
      }
  }
}

void DeclPrinter::VisitClassTemplateSpecializationDecl(
                                           ClassTemplateSpecializationDecl *D) {
  Out << "template<> ";
  VisitCXXRecordDecl(D);
}

void DeclPrinter::VisitClassTemplatePartialSpecializationDecl(
                                    ClassTemplatePartialSpecializationDecl *D) {
  printTemplateParameters(D->getTemplateParameters());
  VisitCXXRecordDecl(D);
}

//----------------------------------------------------------------------------
// Objective-C declarations
//----------------------------------------------------------------------------

void DeclPrinter::PrintObjCMethodType(ASTContext &Ctx,
                                      Decl::ObjCDeclQualifier Quals,
                                      QualType T) {
  Out << '(';
  if (Quals & Decl::ObjCDeclQualifier::OBJC_TQ_In)
    Out << "in ";
  if (Quals & Decl::ObjCDeclQualifier::OBJC_TQ_Inout)
    Out << "inout ";
  if (Quals & Decl::ObjCDeclQualifier::OBJC_TQ_Out)
    Out << "out ";
  if (Quals & Decl::ObjCDeclQualifier::OBJC_TQ_Bycopy)
    Out << "bycopy ";
  if (Quals & Decl::ObjCDeclQualifier::OBJC_TQ_Byref)
    Out << "byref ";
  if (Quals & Decl::ObjCDeclQualifier::OBJC_TQ_Oneway)
    Out << "oneway ";
  if (Quals & Decl::ObjCDeclQualifier::OBJC_TQ_CSNullability) {
    if (auto nullability = AttributedType::stripOuterNullability(T))
      Out << getNullabilitySpelling(*nullability, true) << ' ';
  }

  Out << Ctx.getUnqualifiedObjCPointerType(T).getAsString(Policy);
  Out << ')';
}

void DeclPrinter::PrintObjCTypeParams(ObjCTypeParamList *Params) {
  Out << "<";
  unsigned First = true;
  for (auto *Param : *Params) {
    if (First) {
      First = false;
    } else {
      Out << ", ";
    }

    switch (Param->getVariance()) {
    case ObjCTypeParamVariance::Invariant:
      break;

    case ObjCTypeParamVariance::Covariant:
      Out << "__covariant ";
      break;

    case ObjCTypeParamVariance::Contravariant:
      Out << "__contravariant ";
      break;
    }

    Out << Param->getDeclName();

    if (Param->hasExplicitBound()) {
      Out << " : " << Param->getUnderlyingType().getAsString(Policy);
    }
  }
  Out << ">";
}

void DeclPrinter::VisitObjCMethodDecl(ObjCMethodDecl *OMD) {
  if (OMD->isInstanceMethod())
    Out << "- ";
  else
    Out << "+ ";
  if (!OMD->getReturnType().isNull()) {
    PrintObjCMethodType(OMD->getASTContext(), OMD->getObjCDeclQualifier(),
                        OMD->getReturnType());
  }

  std::string name = OMD->getSelector().getAsString();
  std::string::size_type pos, lastPos = 0;
  for (const auto *PI : OMD->parameters()) {
    // FIXME: selector is missing here!
    pos = name.find_first_of(':', lastPos);
    if (lastPos != 0)
      Out << " ";
    Out << name.substr(lastPos, pos - lastPos) << ':';
    PrintObjCMethodType(OMD->getASTContext(),
                        PI->getObjCDeclQualifier(),
                        PI->getType());
    Out << *PI;
    lastPos = pos + 1;
  }

  if (OMD->param_begin() == OMD->param_end())
    Out << name;

  if (OMD->isVariadic())
      Out << ", ...";

  prettyPrintAttributes(OMD);

  if (OMD->getBody() && !Policy.TerseOutput) {
    Out << ' ';
    printPretty(OMD->getBody(), Out, nullptr, Policy, Indentation, "\n",
                                &Context);
  }
  else if (Policy.PolishForDeclaration)
    Out << ';';
}

void DeclPrinter::VisitObjCImplementationDecl(ObjCImplementationDecl *OID) {
  std::string I = OID->getNameAsString();
  ObjCInterfaceDecl *SID = OID->getSuperClass();

  bool eolnOut = false;
  if (SID)
    Out << "@implementation " << I << " : " << *SID;
  else
    Out << "@implementation " << I;

  if (OID->ivar_size() > 0) {
    Out << "{\n";
    eolnOut = true;
    Indentation += Policy.Indentation;
    for (const auto *I : OID->ivars()) {
      Indent() << I->getASTContext().getUnqualifiedObjCPointerType(I->getType()).
                    getAsString(Policy) << ' ' << *I << ";\n";
    }
    Indentation -= Policy.Indentation;
    Out << "}\n";
  }
  else if (SID || (OID->decls_begin() != OID->decls_end())) {
    Out << "\n";
    eolnOut = true;
  }
  VisitDeclContext(OID, false);
  if (!eolnOut)
    Out << "\n";
  Out << "@end";
}

void DeclPrinter::VisitObjCInterfaceDecl(ObjCInterfaceDecl *OID) {
  std::string I = OID->getNameAsString();
  ObjCInterfaceDecl *SID = OID->getSuperClass();

  if (!OID->isThisDeclarationADefinition()) {
    Out << "@class " << I;

    if (auto *TypeParams = OID->getTypeParamListAsWritten()) {
      PrintObjCTypeParams(TypeParams);
    }

    Out << ";";
    return;
  }
  bool eolnOut = false;
  if (OID->hasAttrs()) {
    prettyPrintAttributes(OID);
    Out << "\n";
  }

  Out << "@interface " << I;

  if (auto *TypeParams = OID->getTypeParamListAsWritten()) {
    PrintObjCTypeParams(TypeParams);
  }

  if (SID)
    Out << " : " << QualType(OID->getSuperClassType(), 0).getAsString(Policy);

  // Protocols?
  const ObjCList<ObjCProtocolDecl> &Protocols = OID->getReferencedProtocols();
  if (!Protocols.empty()) {
    for (ObjCList<ObjCProtocolDecl>::iterator I = Protocols.begin(),
         E = Protocols.end(); I != E; ++I)
      Out << (I == Protocols.begin() ? '<' : ',') << **I;
    Out << "> ";
  }

  if (OID->ivar_size() > 0) {
    Out << "{\n";
    eolnOut = true;
    Indentation += Policy.Indentation;
    for (const auto *I : OID->ivars()) {
      Indent() << I->getASTContext()
                      .getUnqualifiedObjCPointerType(I->getType())
                      .getAsString(Policy) << ' ' << *I << ";\n";
    }
    Indentation -= Policy.Indentation;
    Out << "}\n";
  }
  else if (SID || (OID->decls_begin() != OID->decls_end())) {
    Out << "\n";
    eolnOut = true;
  }

  VisitDeclContext(OID, false);
  if (!eolnOut)
    Out << "\n";
  Out << "@end";
  // FIXME: implement the rest...
}

void DeclPrinter::VisitObjCProtocolDecl(ObjCProtocolDecl *PID) {
  if (!PID->isThisDeclarationADefinition()) {
    Out << "@protocol " << *PID << ";\n";
    return;
  }
  // Protocols?
  const ObjCList<ObjCProtocolDecl> &Protocols = PID->getReferencedProtocols();
  if (!Protocols.empty()) {
    Out << "@protocol " << *PID;
    for (ObjCList<ObjCProtocolDecl>::iterator I = Protocols.begin(),
         E = Protocols.end(); I != E; ++I)
      Out << (I == Protocols.begin() ? '<' : ',') << **I;
    Out << ">\n";
  } else
    Out << "@protocol " << *PID << '\n';
  VisitDeclContext(PID, false);
  Out << "@end";
}

void DeclPrinter::VisitObjCCategoryImplDecl(ObjCCategoryImplDecl *PID) {
  Out << "@implementation ";
  if (const auto *CID = PID->getClassInterface())
    Out << *CID;
  else
    Out << "<<error-type>>";
  Out << '(' << *PID << ")\n";

  VisitDeclContext(PID, false);
  Out << "@end";
  // FIXME: implement the rest...
}

void DeclPrinter::VisitObjCCategoryDecl(ObjCCategoryDecl *PID) {
  Out << "@interface ";
  if (const auto *CID = PID->getClassInterface())
    Out << *CID;
  else
    Out << "<<error-type>>";
  if (auto *TypeParams = PID->getTypeParamList()) {
    PrintObjCTypeParams(TypeParams);
  }
  Out << "(" << *PID << ")\n";
  if (PID->ivar_size() > 0) {
    Out << "{\n";
    Indentation += Policy.Indentation;
    for (const auto *I : PID->ivars())
      Indent() << I->getASTContext().getUnqualifiedObjCPointerType(I->getType()).
                    getAsString(Policy) << ' ' << *I << ";\n";
    Indentation -= Policy.Indentation;
    Out << "}\n";
  }

  VisitDeclContext(PID, false);
  Out << "@end";

  // FIXME: implement the rest...
}

void DeclPrinter::VisitObjCCompatibleAliasDecl(ObjCCompatibleAliasDecl *AID) {
  Out << "@compatibility_alias " << *AID
      << ' ' << *AID->getClassInterface() << ";\n";
}

/// PrintObjCPropertyDecl - print a property declaration.
///
/// Print attributes in the following order:
/// - class
/// - nonatomic | atomic
/// - assign | retain | strong | copy | weak | unsafe_unretained
/// - readwrite | readonly
/// - getter & setter
/// - nullability
void DeclPrinter::VisitObjCPropertyDecl(ObjCPropertyDecl *PDecl) {
  if (PDecl->getPropertyImplementation() == ObjCPropertyDecl::Required)
    Out << "@required\n";
  else if (PDecl->getPropertyImplementation() == ObjCPropertyDecl::Optional)
    Out << "@optional\n";

  QualType T = PDecl->getType();

  Out << "@property";
  if (PDecl->getPropertyAttributes() != ObjCPropertyAttribute::kind_noattr) {
    bool first = true;
    Out << "(";
    if (PDecl->getPropertyAttributes() & ObjCPropertyAttribute::kind_class) {
      Out << (first ? "" : ", ") << "class";
      first = false;
    }

    if (PDecl->getPropertyAttributes() & ObjCPropertyAttribute::kind_direct) {
      Out << (first ? "" : ", ") << "direct";
      first = false;
    }

    if (PDecl->getPropertyAttributes() &
        ObjCPropertyAttribute::kind_nonatomic) {
      Out << (first ? "" : ", ") << "nonatomic";
      first = false;
    }
    if (PDecl->getPropertyAttributes() & ObjCPropertyAttribute::kind_atomic) {
      Out << (first ? "" : ", ") << "atomic";
      first = false;
    }

    if (PDecl->getPropertyAttributes() & ObjCPropertyAttribute::kind_assign) {
      Out << (first ? "" : ", ") << "assign";
      first = false;
    }
    if (PDecl->getPropertyAttributes() & ObjCPropertyAttribute::kind_retain) {
      Out << (first ? "" : ", ") << "retain";
      first = false;
    }

    if (PDecl->getPropertyAttributes() & ObjCPropertyAttribute::kind_strong) {
      Out << (first ? "" : ", ") << "strong";
      first = false;
    }
    if (PDecl->getPropertyAttributes() & ObjCPropertyAttribute::kind_copy) {
      Out << (first ? "" : ", ") << "copy";
      first = false;
    }
    if (PDecl->getPropertyAttributes() & ObjCPropertyAttribute::kind_weak) {
      Out << (first ? "" : ", ") << "weak";
      first = false;
    }
    if (PDecl->getPropertyAttributes() &
        ObjCPropertyAttribute::kind_unsafe_unretained) {
      Out << (first ? "" : ", ") << "unsafe_unretained";
      first = false;
    }

    if (PDecl->getPropertyAttributes() &
        ObjCPropertyAttribute::kind_readwrite) {
      Out << (first ? "" : ", ") << "readwrite";
      first = false;
    }
    if (PDecl->getPropertyAttributes() & ObjCPropertyAttribute::kind_readonly) {
      Out << (first ? "" : ", ") << "readonly";
      first = false;
    }

    if (PDecl->getPropertyAttributes() & ObjCPropertyAttribute::kind_getter) {
      Out << (first ? "" : ", ") << "getter = ";
      PDecl->getGetterName().print(Out);
      first = false;
    }
    if (PDecl->getPropertyAttributes() & ObjCPropertyAttribute::kind_setter) {
      Out << (first ? "" : ", ") << "setter = ";
      PDecl->getSetterName().print(Out);
      first = false;
    }

    if (PDecl->getPropertyAttributes() &
        ObjCPropertyAttribute::kind_nullability) {
      if (auto nullability = AttributedType::stripOuterNullability(T)) {
        if (*nullability == NullabilityKind::Unspecified &&
            (PDecl->getPropertyAttributes() &
             ObjCPropertyAttribute::kind_null_resettable)) {
          Out << (first ? "" : ", ") << "null_resettable";
        } else {
          Out << (first ? "" : ", ")
              << getNullabilitySpelling(*nullability, true);
        }
        first = false;
      }
    }

    (void) first; // Silence dead store warning due to idiomatic code.
    Out << ")";
  }
  std::string TypeStr = PDecl->getASTContext().getUnqualifiedObjCPointerType(T).
      getAsString(Policy);
  Out << ' ' << TypeStr;
  if (!StringRef(TypeStr).ends_with("*"))
    Out << ' ';
  Out << *PDecl;
  if (Policy.PolishForDeclaration)
    Out << ';';
}

void DeclPrinter::VisitObjCPropertyImplDecl(ObjCPropertyImplDecl *PID) {
  if (PID->getPropertyImplementation() == ObjCPropertyImplDecl::Synthesize)
    Out << "@synthesize ";
  else
    Out << "@dynamic ";
  Out << *PID->getPropertyDecl();
  if (PID->getPropertyIvarDecl())
    Out << '=' << *PID->getPropertyIvarDecl();
}

void DeclPrinter::VisitUsingDecl(UsingDecl *D) {
  if (!D->isAccessDeclaration())
    Out << "using ";
  if (D->hasTypename())
    Out << "typename ";
  D->getQualifier()->print(Out, Policy);

  // Use the correct record name when the using declaration is used for
  // inheriting constructors.
  for (const auto *Shadow : D->shadows()) {
    if (const auto *ConstructorShadow =
            dyn_cast<ConstructorUsingShadowDecl>(Shadow)) {
      assert(Shadow->getDeclContext() == ConstructorShadow->getDeclContext());
      Out << *ConstructorShadow->getNominatedBaseClass();
      return;
    }
  }
  Out << *D;
}

void DeclPrinter::VisitUsingEnumDecl(UsingEnumDecl *D) {
  Out << "using enum " << D->getEnumDecl();
}

void
DeclPrinter::VisitUnresolvedUsingTypenameDecl(UnresolvedUsingTypenameDecl *D) {
  Out << "using typename ";
  D->getQualifier()->print(Out, Policy);
  Out << D->getDeclName();
}

void DeclPrinter::VisitUnresolvedUsingValueDecl(UnresolvedUsingValueDecl *D) {
  if (!D->isAccessDeclaration())
    Out << "using ";
  D->getQualifier()->print(Out, Policy);
  Out << D->getDeclName();
}

void DeclPrinter::VisitUsingShadowDecl(UsingShadowDecl *D) {
  // ignore
}

void DeclPrinter::VisitOMPThreadPrivateDecl(OMPThreadPrivateDecl *D) {
  Out << "#pragma omp threadprivate";
  if (!D->varlist_empty()) {
    for (OMPThreadPrivateDecl::varlist_iterator I = D->varlist_begin(),
                                                E = D->varlist_end();
                                                I != E; ++I) {
      Out << (I == D->varlist_begin() ? '(' : ',');
      NamedDecl *ND = cast<DeclRefExpr>(*I)->getDecl();
      ND->printQualifiedName(Out);
    }
    Out << ")";
  }
}

void DeclPrinter::VisitHLSLBufferDecl(HLSLBufferDecl *D) {
  if (D->isCBuffer())
    Out << "cbuffer ";
  else
    Out << "tbuffer ";

  Out << *D;

  prettyPrintAttributes(D);

  Out << " {\n";
  VisitDeclContext(D);
  Indent() << "}";
}

void DeclPrinter::VisitOMPAllocateDecl(OMPAllocateDecl *D) {
  Out << "#pragma omp allocate";
  if (!D->varlist_empty()) {
    for (OMPAllocateDecl::varlist_iterator I = D->varlist_begin(),
                                           E = D->varlist_end();
         I != E; ++I) {
      Out << (I == D->varlist_begin() ? '(' : ',');
      NamedDecl *ND = cast<DeclRefExpr>(*I)->getDecl();
      ND->printQualifiedName(Out);
    }
    Out << ")";
  }
  if (!D->clauselist_empty()) {
    OMPClausePrinter Printer(Out, Policy);
    for (OMPClause *C : D->clauselists()) {
      Out << " ";
      Printer.Visit(C);
    }
  }
}

void DeclPrinter::VisitOMPRequiresDecl(OMPRequiresDecl *D) {
  Out << "#pragma omp requires ";
  if (!D->clauselist_empty()) {
    OMPClausePrinter Printer(Out, Policy);
    for (auto I = D->clauselist_begin(), E = D->clauselist_end(); I != E; ++I)
      Printer.Visit(*I);
  }
}

void DeclPrinter::VisitOMPDeclareReductionDecl(OMPDeclareReductionDecl *D) {
  if (!D->isInvalidDecl()) {
    Out << "#pragma omp declare reduction (";
    if (D->getDeclName().getNameKind() == DeclarationName::CXXOperatorName) {
      const char *OpName =
          getOperatorSpelling(D->getDeclName().getCXXOverloadedOperator());
      assert(OpName && "not an overloaded operator");
      Out << OpName;
    } else {
      assert(D->getDeclName().isIdentifier());
      D->printName(Out, Policy);
    }
    Out << " : ";
    D->getType().print(Out, Policy);
    Out << " : ";
    printPretty(D->getCombiner(), Out, nullptr, Policy, 0, "\n", &Context);
    Out << ")";
    if (auto *Init = D->getInitializer()) {
      Out << " initializer(";
      switch (D->getInitializerKind()) {
      case OMPDeclareReductionInitKind::Direct:
        Out << "omp_priv(";
        break;
      case OMPDeclareReductionInitKind::Copy:
        Out << "omp_priv = ";
        break;
      case OMPDeclareReductionInitKind::Call:
        break;
      }
      printPretty(Init, Out, nullptr, Policy, 0, "\n", &Context);
      if (D->getInitializerKind() == OMPDeclareReductionInitKind::Direct)
        Out << ")";
      Out << ")";
    }
  }
}

void DeclPrinter::VisitOMPDeclareMapperDecl(OMPDeclareMapperDecl *D) {
  if (!D->isInvalidDecl()) {
    Out << "#pragma omp declare mapper (";
    D->printName(Out, Policy);
    Out << " : ";
    D->getType().print(Out, Policy);
    Out << " ";
    Out << D->getVarName();
    Out << ")";
    if (!D->clauselist_empty()) {
      OMPClausePrinter Printer(Out, Policy);
      for (auto *C : D->clauselists()) {
        Out << " ";
        Printer.Visit(C);
      }
    }
  }
}

void DeclPrinter::VisitOMPCapturedExprDecl(OMPCapturedExprDecl *D) {
  printPretty(D->getInit(), Out, nullptr, Policy, Indentation, "\n", &Context);
}

void DeclPrinter::VisitTemplateTypeParmDecl(const TemplateTypeParmDecl *TTP) {
  if (const TypeConstraint *TC = TTP->getTypeConstraint())
    TC->print(Out, Policy);
  else if (TTP->wasDeclaredWithTypename())
    Out << "typename";
  else
    Out << "class";

  if (TTP->isParameterPack())
    Out << " ...";
  else if (TTP->getDeclName())
    Out << ' ';

  if (TTP->getDeclName()) {
    if (Policy.CleanUglifiedParameters && TTP->getIdentifier())
      Out << TTP->getIdentifier()->deuglifiedName();
    else
      Out << TTP->getDeclName();
  }

  if (TTP->hasDefaultArgument()) {
    Out << " = ";
    TTP->getDefaultArgument().getArgument().print(Policy, Out,
                                                  /*IncludeType=*/false);
  }
}

void DeclPrinter::VisitNonTypeTemplateParmDecl(
    const NonTypeTemplateParmDecl *NTTP) {
  StringRef Name;
  if (IdentifierInfo *II = NTTP->getIdentifier())
    Name =
        Policy.CleanUglifiedParameters ? II->deuglifiedName() : II->getName();
  printDeclType(NTTP->getType(), Name, NTTP->isParameterPack());

  if (NTTP->hasDefaultArgument()) {
    Out << " = ";
    NTTP->getDefaultArgument().getArgument().print(Policy, Out,
                                                   /*IncludeType=*/false);
  }
}


//===----------------------------------------------------------------------===//
//  Stmt printing methods.
//===----------------------------------------------------------------------===//

/// PrintRawCompoundStmt - Print a compound stmt without indenting the {, and
/// with no newline after the }.
void DeclPrinter::PrintRawCompoundStmt(CompoundStmt *Node) {
  assert(Node && "Compound statement cannot be null");
  Out << "{" << NL;
  PrintFPPragmas(Node);
  for (auto *I : Node->body())
    PrintStmt(I);

  Indent() << "}";
}

void DeclPrinter::PrintFPPragmas(CompoundStmt *S) {
  if (!S->hasStoredFPFeatures())
    return;
  FPOptionsOverride FPO = S->getStoredFPFeatures();
  bool FEnvAccess = false;
  if (FPO.hasAllowFEnvAccessOverride()) {
    FEnvAccess = FPO.getAllowFEnvAccessOverride();
    Indent() << "#pragma STDC FENV_ACCESS " << (FEnvAccess ? "ON" : "OFF")
             << NL;
  }
  if (FPO.hasSpecifiedExceptionModeOverride()) {
    LangOptions::FPExceptionModeKind EM =
        FPO.getSpecifiedExceptionModeOverride();
    if (!FEnvAccess || EM != LangOptions::FPE_Strict) {
      Indent() << "#pragma clang fp exceptions(";
      switch (FPO.getSpecifiedExceptionModeOverride()) {
      default:
        break;
      case LangOptions::FPE_Ignore:
        Out << "ignore";
        break;
      case LangOptions::FPE_MayTrap:
        Out << "maytrap";
        break;
      case LangOptions::FPE_Strict:
        Out << "strict";
        break;
      }
      Out << ")\n";
    }
  }
  if (FPO.hasConstRoundingModeOverride()) {
    LangOptions::RoundingMode RM = FPO.getConstRoundingModeOverride();
    Indent() << "#pragma STDC FENV_ROUND ";
    switch (RM) {
    case llvm::RoundingMode::TowardZero:
      Out << "FE_TOWARDZERO";
      break;
    case llvm::RoundingMode::NearestTiesToEven:
      Out << "FE_TONEAREST";
      break;
    case llvm::RoundingMode::TowardPositive:
      Out << "FE_UPWARD";
      break;
    case llvm::RoundingMode::TowardNegative:
      Out << "FE_DOWNWARD";
      break;
    case llvm::RoundingMode::NearestTiesToAway:
      Out << "FE_TONEARESTFROMZERO";
      break;
    case llvm::RoundingMode::Dynamic:
      Out << "FE_DYNAMIC";
      break;
    default:
      llvm_unreachable("Invalid rounding mode");
    }
    Out << NL;
  }
}

void DeclPrinter::PrintRawDecl(Decl *D) {
  print(D, Out, Policy, Indentation, false);
}

void DeclPrinter::PrintRawDeclStmt(const DeclStmt *S) {
  SmallVector<Decl *, 2> Decls(S->decls());
  printGroup(Decls.data(), Decls.size(), Out, Policy, Indentation);
}

void DeclPrinter::VisitNullStmt(NullStmt *Node) {
  Indent() << ";" << NL;
}

void DeclPrinter::VisitDeclStmt(DeclStmt *Node) {
  Indent();
  PrintRawDeclStmt(Node);
  Out << ";" << NL;
}

void DeclPrinter::VisitCompoundStmt(CompoundStmt *Node) {
  Indent();
  PrintRawCompoundStmt(Node);
  Out << "" << NL;
}

void DeclPrinter::VisitCaseStmt(CaseStmt *Node) {
  Indent(-1) << "case ";
  PrintExpr(Node->getLHS());
  if (Node->getRHS()) {
    Out << " ... ";
    PrintExpr(Node->getRHS());
  }
  Out << ":" << NL;

  PrintStmt(Node->getSubStmt(), 0);
}

void DeclPrinter::VisitDefaultStmt(DefaultStmt *Node) {
  Indent(-1) << "default:" << NL;
  PrintStmt(Node->getSubStmt(), 0);
}

void DeclPrinter::VisitLabelStmt(LabelStmt *Node) {
  Indent(-1) << Node->getName() << ":" << NL;
  PrintStmt(Node->getSubStmt(), 0);
}

void DeclPrinter::VisitAttributedStmt(AttributedStmt *Node) {
  llvm::ArrayRef<const Attr *> Attrs = Node->getAttrs();
  for (const auto *Attr : Attrs) {
    Attr->printPretty(Out, Policy);
    if (Attr != Attrs.back())
      Out << ' ';
  }

  PrintStmt(Node->getSubStmt(), 0);
}

void DeclPrinter::PrintRawIfStmt(IfStmt *If) {
  if (If->isConsteval()) {
    Out << "if ";
    if (If->isNegatedConsteval())
      Out << "!";
    Out << "consteval";
    Out << NL;
    PrintStmt(If->getThen());
    if (Stmt *Else = If->getElse()) {
      Indent();
      Out << "else";
      PrintStmt(Else);
      Out << NL;
    }
    return;
  }

  Out << "if (";
  if (If->getInit())
    PrintInitStmt(If->getInit(), 4);
  if (const DeclStmt *DS = If->getConditionVariableDeclStmt())
    PrintRawDeclStmt(DS);
  else
    PrintExpr(If->getCond());
  Out << ')';

  if (auto *CS = dyn_cast<CompoundStmt>(If->getThen())) {
    Out << ' ';
    PrintRawCompoundStmt(CS);
    Out << (If->getElse() ? " " : NL);
  } else {
    Out << NL;
    PrintStmt(If->getThen());
    if (If->getElse()) Indent();
  }

  if (Stmt *Else = If->getElse()) {
    Out << "else";

    if (auto *CS = dyn_cast<CompoundStmt>(Else)) {
      Out << ' ';
      PrintRawCompoundStmt(CS);
      Out << NL;
    } else if (auto *ElseIf = dyn_cast<IfStmt>(Else)) {
      Out << ' ';
      PrintRawIfStmt(ElseIf);
    } else {
      Out << NL;
      PrintStmt(If->getElse());
    }
  }
}

void DeclPrinter::VisitIfStmt(IfStmt *If) {
  Indent();
  PrintRawIfStmt(If);
}

void DeclPrinter::VisitSwitchStmt(SwitchStmt *Node) {
  Indent() << "switch (";
  if (Node->getInit())
    PrintInitStmt(Node->getInit(), 8);
  if (const DeclStmt *DS = Node->getConditionVariableDeclStmt())
    PrintRawDeclStmt(DS);
  else
    PrintExpr(Node->getCond());
  Out << ")";
  PrintControlledStmt(Node->getBody());
}

void DeclPrinter::VisitWhileStmt(WhileStmt *Node) {
  Indent() << "while (";
  if (const DeclStmt *DS = Node->getConditionVariableDeclStmt())
    PrintRawDeclStmt(DS);
  else
    PrintExpr(Node->getCond());
  Out << ")" << NL;
  PrintStmt(Node->getBody());
}

void DeclPrinter::VisitDoStmt(DoStmt *Node) {
  Indent() << "do ";
  if (auto *CS = dyn_cast<CompoundStmt>(Node->getBody())) {
    PrintRawCompoundStmt(CS);
    Out << " ";
  } else {
    Out << NL;
    PrintStmt(Node->getBody());
    Indent();
  }

  Out << "while (";
  PrintExpr(Node->getCond());
  Out << ");" << NL;
}

void DeclPrinter::VisitForStmt(ForStmt *Node) {
  Indent() << "for (";
  if (Node->getInit())
    PrintInitStmt(Node->getInit(), 5);
  else
    Out << (Node->getCond() ? "; " : ";");
  if (const DeclStmt *DS = Node->getConditionVariableDeclStmt())
    PrintRawDeclStmt(DS);
  else if (Node->getCond())
    PrintExpr(Node->getCond());
  Out << ";";
  if (Node->getInc()) {
    Out << " ";
    PrintExpr(Node->getInc());
  }
  Out << ")";
  PrintControlledStmt(Node->getBody());
}

void DeclPrinter::VisitObjCForCollectionStmt(ObjCForCollectionStmt *Node) {
  Indent() << "for (";
  if (auto *DS = dyn_cast<DeclStmt>(Node->getElement()))
    PrintRawDeclStmt(DS);
  else
    PrintExpr(cast<Expr>(Node->getElement()));
  Out << " in ";
  PrintExpr(Node->getCollection());
  Out << ")";
  PrintControlledStmt(Node->getBody());
}

void DeclPrinter::VisitCXXForRangeStmt(CXXForRangeStmt *Node) {
  Indent() << "for (";
  if (Node->getInit())
    PrintInitStmt(Node->getInit(), 5);
  PrintingPolicy SubPolicy(Policy);
  SubPolicy.SuppressInitializers = true;
  print(Node->getLoopVariable(), Out, SubPolicy, Indentation, false);
  Out << " : ";
  PrintExpr(Node->getRangeInit());
  Out << ")";
  PrintControlledStmt(Node->getBody());
}

void DeclPrinter::VisitMSDependentExistsStmt(MSDependentExistsStmt *Node) {
  Indent();
  if (Node->isIfExists())
    Out << "__if_exists (";
  else
    Out << "__if_not_exists (";

  if (NestedNameSpecifier *Qualifier
        = Node->getQualifierLoc().getNestedNameSpecifier())
    Qualifier->print(Out, Policy);

  Out << Node->getNameInfo() << ") ";

  PrintRawCompoundStmt(Node->getSubStmt());
}

void DeclPrinter::VisitGotoStmt(GotoStmt *Node) {
  Indent() << "goto " << Node->getLabel()->getName() << ";";
  if (Policy.IncludeNewlines) Out << NL;
}

void DeclPrinter::VisitIndirectGotoStmt(IndirectGotoStmt *Node) {
  Indent() << "goto *";
  PrintExpr(Node->getTarget());
  Out << ";";
  if (Policy.IncludeNewlines) Out << NL;
}

void DeclPrinter::VisitContinueStmt(ContinueStmt *Node) {
  Indent() << "continue;";
  if (Policy.IncludeNewlines) Out << NL;
}

void DeclPrinter::VisitBreakStmt(BreakStmt *Node) {
  Indent() << "break;";
  if (Policy.IncludeNewlines) Out << NL;
}

void DeclPrinter::VisitReturnStmt(ReturnStmt *Node) {
  Indent() << "return";
  if (Node->getRetValue()) {
    Out << " ";
    PrintExpr(Node->getRetValue());
  }
  Out << ";";
  if (Policy.IncludeNewlines) Out << NL;
}

void DeclPrinter::VisitGCCAsmStmt(GCCAsmStmt *Node) {
  Indent() << "asm ";

  if (Node->isVolatile())
    Out << "volatile ";

  if (Node->isAsmGoto())
    Out << "goto ";

  Out << "(";
  VisitStringLiteral(Node->getAsmString());

  // Outputs
  if (Node->getNumOutputs() != 0 || Node->getNumInputs() != 0 ||
      Node->getNumClobbers() != 0 || Node->getNumLabels() != 0)
    Out << " : ";

  for (unsigned i = 0, e = Node->getNumOutputs(); i != e; ++i) {
    if (i != 0)
      Out << ", ";

    if (!Node->getOutputName(i).empty()) {
      Out << '[';
      Out << Node->getOutputName(i);
      Out << "] ";
    }

    VisitStringLiteral(Node->getOutputConstraintLiteral(i));
    Out << " (";
    Visit(Node->getOutputExpr(i));
    Out << ")";
  }

  // Inputs
  if (Node->getNumInputs() != 0 || Node->getNumClobbers() != 0 ||
      Node->getNumLabels() != 0)
    Out << " : ";

  for (unsigned i = 0, e = Node->getNumInputs(); i != e; ++i) {
    if (i != 0)
      Out << ", ";

    if (!Node->getInputName(i).empty()) {
      Out << '[';
      Out << Node->getInputName(i);
      Out << "] ";
    }

    VisitStringLiteral(Node->getInputConstraintLiteral(i));
    Out << " (";
    Visit(Node->getInputExpr(i));
    Out << ")";
  }

  // Clobbers
  if (Node->getNumClobbers() != 0 || Node->getNumLabels())
    Out << " : ";

  for (unsigned i = 0, e = Node->getNumClobbers(); i != e; ++i) {
    if (i != 0)
      Out << ", ";

    VisitStringLiteral(Node->getClobberStringLiteral(i));
  }

  // Labels
  if (Node->getNumLabels() != 0)
    Out << " : ";

  for (unsigned i = 0, e = Node->getNumLabels(); i != e; ++i) {
    if (i != 0)
      Out << ", ";
    Out << Node->getLabelName(i);
  }

  Out << ");";
  if (Policy.IncludeNewlines) Out << NL;
}

void DeclPrinter::VisitMSAsmStmt(MSAsmStmt *Node) {
  // FIXME: Implement MS style inline asm statement printer.
  Indent() << "__asm ";
  if (Node->hasBraces())
    Out << "{" << NL;
  Out << Node->getAsmString() << NL;
  if (Node->hasBraces())
    Indent() << "}" << NL;
}

void DeclPrinter::VisitCapturedStmt(CapturedStmt *Node) {
  PrintStmt(Node->getCapturedDecl()->getBody());
}

void DeclPrinter::VisitObjCAtTryStmt(ObjCAtTryStmt *Node) {
  Indent() << "@try";
  if (auto *TS = dyn_cast<CompoundStmt>(Node->getTryBody())) {
    PrintRawCompoundStmt(TS);
    Out << NL;
  }

  for (ObjCAtCatchStmt *catchStmt : Node->catch_stmts()) {
    Indent() << "@catch(";
    if (Decl *DS = catchStmt->getCatchParamDecl())
      PrintRawDecl(DS);
    Out << ")";
    if (auto *CS = dyn_cast<CompoundStmt>(catchStmt->getCatchBody())) {
      PrintRawCompoundStmt(CS);
      Out << NL;
    }
  }

  if (auto *FS = static_cast<ObjCAtFinallyStmt *>(Node->getFinallyStmt())) {
    Indent() << "@finally";
    if (auto *CS = dyn_cast<CompoundStmt>(FS->getFinallyBody())) {
      PrintRawCompoundStmt(CS);
      Out << NL;
    }
  }
}

void DeclPrinter::VisitObjCAtFinallyStmt(ObjCAtFinallyStmt *Node) {
}

void DeclPrinter::VisitObjCAtCatchStmt (ObjCAtCatchStmt *Node) {
  Indent() << "@catch (...) { /* todo */ } " << NL;
}

void DeclPrinter::VisitObjCAtThrowStmt(ObjCAtThrowStmt *Node) {
  Indent() << "@throw";
  if (Node->getThrowExpr()) {
    Out << " ";
    PrintExpr(Node->getThrowExpr());
  }
  Out << ";" << NL;
}

void DeclPrinter::VisitObjCAvailabilityCheckExpr(
    ObjCAvailabilityCheckExpr *Node) {
  Out << "@available(...)";
}

void DeclPrinter::VisitObjCAtSynchronizedStmt(ObjCAtSynchronizedStmt *Node) {
  Indent() << "@synchronized (";
  PrintExpr(Node->getSynchExpr());
  Out << ")";
  PrintRawCompoundStmt(Node->getSynchBody());
  Out << NL;
}

void DeclPrinter::VisitObjCAutoreleasePoolStmt(ObjCAutoreleasePoolStmt *Node) {
  Indent() << "@autoreleasepool";
  PrintRawCompoundStmt(cast<CompoundStmt>(Node->getSubStmt()));
  Out << NL;
}

void DeclPrinter::PrintRawCXXCatchStmt(CXXCatchStmt *Node) {
  Out << "catch (";
  if (Decl *ExDecl = Node->getExceptionDecl())
    PrintRawDecl(ExDecl);
  else
    Out << "...";
  Out << ") ";
  PrintRawCompoundStmt(cast<CompoundStmt>(Node->getHandlerBlock()));
}

void DeclPrinter::VisitCXXCatchStmt(CXXCatchStmt *Node) {
  Indent();
  PrintRawCXXCatchStmt(Node);
  Out << NL;
}

void DeclPrinter::VisitCXXTryStmt(CXXTryStmt *Node) {
  Indent() << "try ";
  PrintRawCompoundStmt(Node->getTryBlock());
  for (unsigned i = 0, e = Node->getNumHandlers(); i < e; ++i) {
    Out << " ";
    PrintRawCXXCatchStmt(Node->getHandler(i));
  }
  Out << NL;
}

void DeclPrinter::VisitSEHTryStmt(SEHTryStmt *Node) {
  Indent() << (Node->getIsCXXTry() ? "try " : "__try ");
  PrintRawCompoundStmt(Node->getTryBlock());
  SEHExceptStmt *E = Node->getExceptHandler();
  SEHFinallyStmt *F = Node->getFinallyHandler();
  if(E)
    PrintRawSEHExceptHandler(E);
  else {
    assert(F && "Must have a finally block...");
    PrintRawSEHFinallyStmt(F);
  }
  Out << NL;
}

void DeclPrinter::PrintRawSEHFinallyStmt(SEHFinallyStmt *Node) {
  Out << "__finally ";
  PrintRawCompoundStmt(Node->getBlock());
  Out << NL;
}

void DeclPrinter::PrintRawSEHExceptHandler(SEHExceptStmt *Node) {
  Out << "__except (";
  VisitExpr(Node->getFilterExpr());
  Out << ")" << NL;
  PrintRawCompoundStmt(Node->getBlock());
  Out << NL;
}

void DeclPrinter::VisitSEHExceptStmt(SEHExceptStmt *Node) {
  Indent();
  PrintRawSEHExceptHandler(Node);
  Out << NL;
}

void DeclPrinter::VisitSEHFinallyStmt(SEHFinallyStmt *Node) {
  Indent();
  PrintRawSEHFinallyStmt(Node);
  Out << NL;
}

void DeclPrinter::VisitSEHLeaveStmt(SEHLeaveStmt *Node) {
  Indent() << "__leave;";
  if (Policy.IncludeNewlines) Out << NL;
}

//===----------------------------------------------------------------------===//
//  OpenMP directives printing methods
//===----------------------------------------------------------------------===//

void DeclPrinter::VisitOMPCanonicalLoop(OMPCanonicalLoop *Node) {
  PrintStmt(Node->getLoopStmt());
}

void DeclPrinter::PrintOMPExecutableDirective(OMPExecutableDirective *S,
                                              bool ForceNoStmt) {
  OMPClausePrinter Printer(Out, Policy);
  ArrayRef<OMPClause *> Clauses = S->clauses();
  for (auto *Clause : Clauses)
    if (Clause && !Clause->isImplicit()) {
      Out << ' ';
      Printer.Visit(Clause);
    }
  Out << NL;
  if (!ForceNoStmt && S->hasAssociatedStmt())
    PrintStmt(S->getRawStmt());
}

void DeclPrinter::VisitOMPMetaDirective(OMPMetaDirective *Node) {
  Indent() << "#pragma omp metadirective";
  PrintOMPExecutableDirective(Node);
}

void DeclPrinter::VisitOMPParallelDirective(OMPParallelDirective *Node) {
  Indent() << "#pragma omp parallel";
  PrintOMPExecutableDirective(Node);
}

void DeclPrinter::VisitOMPSimdDirective(OMPSimdDirective *Node) {
  Indent() << "#pragma omp simd";
  PrintOMPExecutableDirective(Node);
}

void DeclPrinter::VisitOMPTileDirective(OMPTileDirective *Node) {
  Indent() << "#pragma omp tile";
  PrintOMPExecutableDirective(Node);
}

void DeclPrinter::VisitOMPUnrollDirective(OMPUnrollDirective *Node) {
  Indent() << "#pragma omp unroll";
  PrintOMPExecutableDirective(Node);
}

void DeclPrinter::VisitOMPReverseDirective(OMPReverseDirective *Node) {
  Indent() << "#pragma omp reverse";
  PrintOMPExecutableDirective(Node);
}

void DeclPrinter::VisitOMPInterchangeDirective(OMPInterchangeDirective *Node) {
  Indent() << "#pragma omp interchange";
  PrintOMPExecutableDirective(Node);
}

void DeclPrinter::VisitOMPForDirective(OMPForDirective *Node) {
  Indent() << "#pragma omp for";
  PrintOMPExecutableDirective(Node);
}

void DeclPrinter::VisitOMPForSimdDirective(OMPForSimdDirective *Node) {
  Indent() << "#pragma omp for simd";
  PrintOMPExecutableDirective(Node);
}

void DeclPrinter::VisitOMPSectionsDirective(OMPSectionsDirective *Node) {
  Indent() << "#pragma omp sections";
  PrintOMPExecutableDirective(Node);
}

void DeclPrinter::VisitOMPSectionDirective(OMPSectionDirective *Node) {
  Indent() << "#pragma omp section";
  PrintOMPExecutableDirective(Node);
}

void DeclPrinter::VisitOMPScopeDirective(OMPScopeDirective *Node) {
  Indent() << "#pragma omp scope";
  PrintOMPExecutableDirective(Node);
}

void DeclPrinter::VisitOMPSingleDirective(OMPSingleDirective *Node) {
  Indent() << "#pragma omp single";
  PrintOMPExecutableDirective(Node);
}

void DeclPrinter::VisitOMPMasterDirective(OMPMasterDirective *Node) {
  Indent() << "#pragma omp master";
  PrintOMPExecutableDirective(Node);
}

void DeclPrinter::VisitOMPCriticalDirective(OMPCriticalDirective *Node) {
  Indent() << "#pragma omp critical";
  if (Node->getDirectiveName().getName()) {
    Out << " (";
    Node->getDirectiveName().printName(Out, Policy);
    Out << ")";
  }
  PrintOMPExecutableDirective(Node);
}

void DeclPrinter::VisitOMPParallelForDirective(OMPParallelForDirective *Node) {
  Indent() << "#pragma omp parallel for";
  PrintOMPExecutableDirective(Node);
}

void DeclPrinter::VisitOMPParallelForSimdDirective(
    OMPParallelForSimdDirective *Node) {
  Indent() << "#pragma omp parallel for simd";
  PrintOMPExecutableDirective(Node);
}

void DeclPrinter::VisitOMPParallelMasterDirective(
    OMPParallelMasterDirective *Node) {
  Indent() << "#pragma omp parallel master";
  PrintOMPExecutableDirective(Node);
}

void DeclPrinter::VisitOMPParallelMaskedDirective(
    OMPParallelMaskedDirective *Node) {
  Indent() << "#pragma omp parallel masked";
  PrintOMPExecutableDirective(Node);
}

void DeclPrinter::VisitOMPParallelSectionsDirective(
    OMPParallelSectionsDirective *Node) {
  Indent() << "#pragma omp parallel sections";
  PrintOMPExecutableDirective(Node);
}

void DeclPrinter::VisitOMPTaskDirective(OMPTaskDirective *Node) {
  Indent() << "#pragma omp task";
  PrintOMPExecutableDirective(Node);
}

void DeclPrinter::VisitOMPTaskyieldDirective(OMPTaskyieldDirective *Node) {
  Indent() << "#pragma omp taskyield";
  PrintOMPExecutableDirective(Node);
}

void DeclPrinter::VisitOMPBarrierDirective(OMPBarrierDirective *Node) {
  Indent() << "#pragma omp barrier";
  PrintOMPExecutableDirective(Node);
}

void DeclPrinter::VisitOMPTaskwaitDirective(OMPTaskwaitDirective *Node) {
  Indent() << "#pragma omp taskwait";
  PrintOMPExecutableDirective(Node);
}

void DeclPrinter::VisitOMPErrorDirective(OMPErrorDirective *Node) {
  Indent() << "#pragma omp error";
  PrintOMPExecutableDirective(Node);
}

void DeclPrinter::VisitOMPTaskgroupDirective(OMPTaskgroupDirective *Node) {
  Indent() << "#pragma omp taskgroup";
  PrintOMPExecutableDirective(Node);
}

void DeclPrinter::VisitOMPFlushDirective(OMPFlushDirective *Node) {
  Indent() << "#pragma omp flush";
  PrintOMPExecutableDirective(Node);
}

void DeclPrinter::VisitOMPDepobjDirective(OMPDepobjDirective *Node) {
  Indent() << "#pragma omp depobj";
  PrintOMPExecutableDirective(Node);
}

void DeclPrinter::VisitOMPScanDirective(OMPScanDirective *Node) {
  Indent() << "#pragma omp scan";
  PrintOMPExecutableDirective(Node);
}

void DeclPrinter::VisitOMPOrderedDirective(OMPOrderedDirective *Node) {
  Indent() << "#pragma omp ordered";
  PrintOMPExecutableDirective(Node, Node->hasClausesOfKind<OMPDependClause>());
}

void DeclPrinter::VisitOMPAtomicDirective(OMPAtomicDirective *Node) {
  Indent() << "#pragma omp atomic";
  PrintOMPExecutableDirective(Node);
}

void DeclPrinter::VisitOMPTargetDirective(OMPTargetDirective *Node) {
  Indent() << "#pragma omp target";
  PrintOMPExecutableDirective(Node);
}

void DeclPrinter::VisitOMPTargetDataDirective(OMPTargetDataDirective *Node) {
  Indent() << "#pragma omp target data";
  PrintOMPExecutableDirective(Node);
}

void DeclPrinter::VisitOMPTargetEnterDataDirective(
    OMPTargetEnterDataDirective *Node) {
  Indent() << "#pragma omp target enter data";
  PrintOMPExecutableDirective(Node, /*ForceNoStmt=*/true);
}

void DeclPrinter::VisitOMPTargetExitDataDirective(
    OMPTargetExitDataDirective *Node) {
  Indent() << "#pragma omp target exit data";
  PrintOMPExecutableDirective(Node, /*ForceNoStmt=*/true);
}

void DeclPrinter::VisitOMPTargetParallelDirective(
    OMPTargetParallelDirective *Node) {
  Indent() << "#pragma omp target parallel";
  PrintOMPExecutableDirective(Node);
}

void DeclPrinter::VisitOMPTargetParallelForDirective(
    OMPTargetParallelForDirective *Node) {
  Indent() << "#pragma omp target parallel for";
  PrintOMPExecutableDirective(Node);
}

void DeclPrinter::VisitOMPTeamsDirective(OMPTeamsDirective *Node) {
  Indent() << "#pragma omp teams";
  PrintOMPExecutableDirective(Node);
}

void DeclPrinter::VisitOMPCancellationPointDirective(
    OMPCancellationPointDirective *Node) {
  Indent() << "#pragma omp cancellation point "
           << getOpenMPDirectiveName(Node->getCancelRegion());
  PrintOMPExecutableDirective(Node);
}

void DeclPrinter::VisitOMPCancelDirective(OMPCancelDirective *Node) {
  Indent() << "#pragma omp cancel "
           << getOpenMPDirectiveName(Node->getCancelRegion());
  PrintOMPExecutableDirective(Node);
}

void DeclPrinter::VisitOMPTaskLoopDirective(OMPTaskLoopDirective *Node) {
  Indent() << "#pragma omp taskloop";
  PrintOMPExecutableDirective(Node);
}

void DeclPrinter::VisitOMPTaskLoopSimdDirective(
    OMPTaskLoopSimdDirective *Node) {
  Indent() << "#pragma omp taskloop simd";
  PrintOMPExecutableDirective(Node);
}

void DeclPrinter::VisitOMPMasterTaskLoopDirective(
    OMPMasterTaskLoopDirective *Node) {
  Indent() << "#pragma omp master taskloop";
  PrintOMPExecutableDirective(Node);
}

void DeclPrinter::VisitOMPMaskedTaskLoopDirective(
    OMPMaskedTaskLoopDirective *Node) {
  Indent() << "#pragma omp masked taskloop";
  PrintOMPExecutableDirective(Node);
}

void DeclPrinter::VisitOMPMasterTaskLoopSimdDirective(
    OMPMasterTaskLoopSimdDirective *Node) {
  Indent() << "#pragma omp master taskloop simd";
  PrintOMPExecutableDirective(Node);
}

void DeclPrinter::VisitOMPMaskedTaskLoopSimdDirective(
    OMPMaskedTaskLoopSimdDirective *Node) {
  Indent() << "#pragma omp masked taskloop simd";
  PrintOMPExecutableDirective(Node);
}

void DeclPrinter::VisitOMPParallelMasterTaskLoopDirective(
    OMPParallelMasterTaskLoopDirective *Node) {
  Indent() << "#pragma omp parallel master taskloop";
  PrintOMPExecutableDirective(Node);
}

void DeclPrinter::VisitOMPParallelMaskedTaskLoopDirective(
    OMPParallelMaskedTaskLoopDirective *Node) {
  Indent() << "#pragma omp parallel masked taskloop";
  PrintOMPExecutableDirective(Node);
}

void DeclPrinter::VisitOMPParallelMasterTaskLoopSimdDirective(
    OMPParallelMasterTaskLoopSimdDirective *Node) {
  Indent() << "#pragma omp parallel master taskloop simd";
  PrintOMPExecutableDirective(Node);
}

void DeclPrinter::VisitOMPParallelMaskedTaskLoopSimdDirective(
    OMPParallelMaskedTaskLoopSimdDirective *Node) {
  Indent() << "#pragma omp parallel masked taskloop simd";
  PrintOMPExecutableDirective(Node);
}

void DeclPrinter::VisitOMPDistributeDirective(OMPDistributeDirective *Node) {
  Indent() << "#pragma omp distribute";
  PrintOMPExecutableDirective(Node);
}

void DeclPrinter::VisitOMPTargetUpdateDirective(
    OMPTargetUpdateDirective *Node) {
  Indent() << "#pragma omp target update";
  PrintOMPExecutableDirective(Node, /*ForceNoStmt=*/true);
}

void DeclPrinter::VisitOMPDistributeParallelForDirective(
    OMPDistributeParallelForDirective *Node) {
  Indent() << "#pragma omp distribute parallel for";
  PrintOMPExecutableDirective(Node);
}

void DeclPrinter::VisitOMPDistributeParallelForSimdDirective(
    OMPDistributeParallelForSimdDirective *Node) {
  Indent() << "#pragma omp distribute parallel for simd";
  PrintOMPExecutableDirective(Node);
}

void DeclPrinter::VisitOMPDistributeSimdDirective(
    OMPDistributeSimdDirective *Node) {
  Indent() << "#pragma omp distribute simd";
  PrintOMPExecutableDirective(Node);
}

void DeclPrinter::VisitOMPTargetParallelForSimdDirective(
    OMPTargetParallelForSimdDirective *Node) {
  Indent() << "#pragma omp target parallel for simd";
  PrintOMPExecutableDirective(Node);
}

void DeclPrinter::VisitOMPTargetSimdDirective(OMPTargetSimdDirective *Node) {
  Indent() << "#pragma omp target simd";
  PrintOMPExecutableDirective(Node);
}

void DeclPrinter::VisitOMPTeamsDistributeDirective(
    OMPTeamsDistributeDirective *Node) {
  Indent() << "#pragma omp teams distribute";
  PrintOMPExecutableDirective(Node);
}

void DeclPrinter::VisitOMPTeamsDistributeSimdDirective(
    OMPTeamsDistributeSimdDirective *Node) {
  Indent() << "#pragma omp teams distribute simd";
  PrintOMPExecutableDirective(Node);
}

void DeclPrinter::VisitOMPTeamsDistributeParallelForSimdDirective(
    OMPTeamsDistributeParallelForSimdDirective *Node) {
  Indent() << "#pragma omp teams distribute parallel for simd";
  PrintOMPExecutableDirective(Node);
}

void DeclPrinter::VisitOMPTeamsDistributeParallelForDirective(
    OMPTeamsDistributeParallelForDirective *Node) {
  Indent() << "#pragma omp teams distribute parallel for";
  PrintOMPExecutableDirective(Node);
}

void DeclPrinter::VisitOMPTargetTeamsDirective(OMPTargetTeamsDirective *Node) {
  Indent() << "#pragma omp target teams";
  PrintOMPExecutableDirective(Node);
}

void DeclPrinter::VisitOMPTargetTeamsDistributeDirective(
    OMPTargetTeamsDistributeDirective *Node) {
  Indent() << "#pragma omp target teams distribute";
  PrintOMPExecutableDirective(Node);
}

void DeclPrinter::VisitOMPTargetTeamsDistributeParallelForDirective(
    OMPTargetTeamsDistributeParallelForDirective *Node) {
  Indent() << "#pragma omp target teams distribute parallel for";
  PrintOMPExecutableDirective(Node);
}

void DeclPrinter::VisitOMPTargetTeamsDistributeParallelForSimdDirective(
    OMPTargetTeamsDistributeParallelForSimdDirective *Node) {
  Indent() << "#pragma omp target teams distribute parallel for simd";
  PrintOMPExecutableDirective(Node);
}

void DeclPrinter::VisitOMPTargetTeamsDistributeSimdDirective(
    OMPTargetTeamsDistributeSimdDirective *Node) {
  Indent() << "#pragma omp target teams distribute simd";
  PrintOMPExecutableDirective(Node);
}

void DeclPrinter::VisitOMPInteropDirective(OMPInteropDirective *Node) {
  Indent() << "#pragma omp interop";
  PrintOMPExecutableDirective(Node);
}

void DeclPrinter::VisitOMPDispatchDirective(OMPDispatchDirective *Node) {
  Indent() << "#pragma omp dispatch";
  PrintOMPExecutableDirective(Node);
}

void DeclPrinter::VisitOMPMaskedDirective(OMPMaskedDirective *Node) {
  Indent() << "#pragma omp masked";
  PrintOMPExecutableDirective(Node);
}

void DeclPrinter::VisitOMPGenericLoopDirective(OMPGenericLoopDirective *Node) {
  Indent() << "#pragma omp loop";
  PrintOMPExecutableDirective(Node);
}

void DeclPrinter::VisitOMPTeamsGenericLoopDirective(
    OMPTeamsGenericLoopDirective *Node) {
  Indent() << "#pragma omp teams loop";
  PrintOMPExecutableDirective(Node);
}

void DeclPrinter::VisitOMPTargetTeamsGenericLoopDirective(
    OMPTargetTeamsGenericLoopDirective *Node) {
  Indent() << "#pragma omp target teams loop";
  PrintOMPExecutableDirective(Node);
}

void DeclPrinter::VisitOMPParallelGenericLoopDirective(
    OMPParallelGenericLoopDirective *Node) {
  Indent() << "#pragma omp parallel loop";
  PrintOMPExecutableDirective(Node);
}

void DeclPrinter::VisitOMPTargetParallelGenericLoopDirective(
    OMPTargetParallelGenericLoopDirective *Node) {
  Indent() << "#pragma omp target parallel loop";
  PrintOMPExecutableDirective(Node);
}

//===----------------------------------------------------------------------===//
//  OpenACC construct printing methods
//===----------------------------------------------------------------------===//
void DeclPrinter::VisitOpenACCComputeConstruct(OpenACCComputeConstruct *S) {
  Indent() << "#pragma acc " << S->getDirectiveKind();

  if (!S->clauses().empty()) {
    Out << ' ';
    OpenACCClausePrinter Printer(Out, Policy);
    Printer.VisitClauseList(S->clauses());
  }
  Out << '\n';

  PrintStmt(S->getStructuredBlock());
}

void DeclPrinter::VisitOpenACCLoopConstruct(OpenACCLoopConstruct *S) {
  Indent() << "#pragma acc loop";

  if (!S->clauses().empty()) {
    Out << ' ';
    OpenACCClausePrinter Printer(Out, Policy);
    Printer.VisitClauseList(S->clauses());
  }
  Out << '\n';

  PrintStmt(S->getLoop());
}

//===----------------------------------------------------------------------===//
//  Expr printing methods.
//===----------------------------------------------------------------------===//

void DeclPrinter::VisitSourceLocExpr(SourceLocExpr *Node) {
  Out << Node->getBuiltinStr() << "()";
}

void DeclPrinter::VisitEmbedExpr(EmbedExpr *Node) {
  llvm::report_fatal_error("Not implemented");
}

void DeclPrinter::VisitConstantExpr(ConstantExpr *Node) {
  PrintExpr(Node->getSubExpr());
}

void DeclPrinter::VisitDeclRefExpr(DeclRefExpr *Node) {
  if (const auto *OCED = dyn_cast<OMPCapturedExprDecl>(Node->getDecl())) {
    printPretty(OCED->getInit()->IgnoreImpCasts(), Out, nullptr, Policy, 0, "\n", nullptr);
    return;
  }
  if (const auto *TPOD = dyn_cast<TemplateParamObjectDecl>(Node->getDecl())) {
    TPOD->printAsExpr(Out, Policy);
    return;
  }
  if (NestedNameSpecifier *Qualifier = Node->getQualifier())
    Qualifier->print(Out, Policy);
  if (Node->hasTemplateKeyword())
    Out << "template ";
  if (Policy.CleanUglifiedParameters &&
      isa<ParmVarDecl, NonTypeTemplateParmDecl>(Node->getDecl()) &&
      Node->getDecl()->getIdentifier())
    Out << Node->getDecl()->getIdentifier()->deuglifiedName();
  else
    Node->getNameInfo().printName(Out, Policy);
  if (Node->hasExplicitTemplateArgs()) {
    const TemplateParameterList *TPL = nullptr;
    if (!Node->hadMultipleCandidates())
      if (auto *TD = dyn_cast<TemplateDecl>(Node->getDecl()))
        TPL = TD->getTemplateParameters();
    printTemplateArgumentList(Out, Node->template_arguments(), Policy, TPL);
  }
}

void DeclPrinter::VisitDependentScopeDeclRefExpr(
                                           DependentScopeDeclRefExpr *Node) {
  if (NestedNameSpecifier *Qualifier = Node->getQualifier())
    Qualifier->print(Out, Policy);
  if (Node->hasTemplateKeyword())
    Out << "template ";
  Out << Node->getNameInfo();
  if (Node->hasExplicitTemplateArgs())
    printTemplateArgumentList(Out, Node->template_arguments(), Policy);
}

void DeclPrinter::VisitUnresolvedLookupExpr(UnresolvedLookupExpr *Node) {
  if (Node->getQualifier())
    Node->getQualifier()->print(Out, Policy);
  if (Node->hasTemplateKeyword())
    Out << "template ";
  Out << Node->getNameInfo();
  if (Node->hasExplicitTemplateArgs())
    printTemplateArgumentList(Out, Node->template_arguments(), Policy);
}

static bool isImplicitSelf(const Expr *E) {
  if (const auto *DRE = dyn_cast<DeclRefExpr>(E)) {
    if (const auto *PD = dyn_cast<ImplicitParamDecl>(DRE->getDecl())) {
      if (PD->getParameterKind() == ImplicitParamKind::ObjCSelf &&
          DRE->getBeginLoc().isInvalid())
        return true;
    }
  }
  return false;
}

void DeclPrinter::VisitObjCIvarRefExpr(ObjCIvarRefExpr *Node) {
  if (Node->getBase()) {
    if (!Policy.SuppressImplicitBase ||
        !isImplicitSelf(Node->getBase()->IgnoreImpCasts())) {
      PrintExpr(Node->getBase());
      Out << (Node->isArrow() ? "->" : ".");
    }
  }
  Out << *Node->getDecl();
}

void DeclPrinter::VisitObjCPropertyRefExpr(ObjCPropertyRefExpr *Node) {
  if (Node->isSuperReceiver())
    Out << "super.";
  else if (Node->isObjectReceiver() && Node->getBase()) {
    PrintExpr(Node->getBase());
    Out << ".";
  } else if (Node->isClassReceiver() && Node->getClassReceiver()) {
    Out << Node->getClassReceiver()->getName() << ".";
  }

  if (Node->isImplicitProperty()) {
    if (const auto *Getter = Node->getImplicitPropertyGetter())
      Getter->getSelector().print(Out);
    else
      Out << SelectorTable::getPropertyNameFromSetterSelector(
          Node->getImplicitPropertySetter()->getSelector());
  } else
    Out << Node->getExplicitProperty()->getName();
}

void DeclPrinter::VisitObjCSubscriptRefExpr(ObjCSubscriptRefExpr *Node) {
  PrintExpr(Node->getBaseExpr());
  Out << "[";
  PrintExpr(Node->getKeyExpr());
  Out << "]";
}

void DeclPrinter::VisitSYCLUniqueStableNameExpr(
    SYCLUniqueStableNameExpr *Node) {
  Out << "__builtin_sycl_unique_stable_name(";
  Node->getTypeSourceInfo()->getType().print(Out, Policy);
  Out << ")";
}

void DeclPrinter::VisitPredefinedExpr(PredefinedExpr *Node) {
  Out << PredefinedExpr::getIdentKindName(Node->getIdentKind());
}

void DeclPrinter::VisitCharacterLiteral(CharacterLiteral *Node) {
  CharacterLiteral::print(Node->getValue(), Node->getKind(), Out);
}

void DeclPrinter::VisitIntegerLiteral(IntegerLiteral *Node) {
  bool isSigned = Node->getType()->isSignedIntegerType();
  Out << toString(Node->getValue(), 10, isSigned);

  if (isa<BitIntType>(Node->getType())) {
    Out << (isSigned ? "wb" : "uwb");
    return;
  }

  // Emit suffixes.  Integer literals are always a builtin integer type.
  switch (Node->getType()->castAs<BuiltinType>()->getKind()) {
  default: llvm_unreachable("Unexpected type for integer literal!");
  case BuiltinType::Char_S:
  case BuiltinType::Char_U:    Out << "i8"; break;
  case BuiltinType::UChar:     Out << "Ui8"; break;
  case BuiltinType::SChar:     Out << "i8"; break;
  case BuiltinType::Short:     Out << "i16"; break;
  case BuiltinType::UShort:    Out << "Ui16"; break;
  case BuiltinType::Int:       break; // no suffix.
  case BuiltinType::UInt:      Out << 'U'; break;
  case BuiltinType::Long:      Out << 'L'; break;
  case BuiltinType::ULong:     Out << "UL"; break;
  case BuiltinType::LongLong:  Out << "LL"; break;
  case BuiltinType::ULongLong: Out << "ULL"; break;
  case BuiltinType::Int128:
    break; // no suffix.
  case BuiltinType::UInt128:
    break; // no suffix.
  case BuiltinType::WChar_S:
  case BuiltinType::WChar_U:
    break; // no suffix
  }
}

void DeclPrinter::VisitFixedPointLiteral(FixedPointLiteral *Node) {
  Out << Node->getValueAsString(/*Radix=*/10);

  switch (Node->getType()->castAs<BuiltinType>()->getKind()) {
    default: llvm_unreachable("Unexpected type for fixed point literal!");
    case BuiltinType::ShortFract:   Out << "hr"; break;
    case BuiltinType::ShortAccum:   Out << "hk"; break;
    case BuiltinType::UShortFract:  Out << "uhr"; break;
    case BuiltinType::UShortAccum:  Out << "uhk"; break;
    case BuiltinType::Fract:        Out << "r"; break;
    case BuiltinType::Accum:        Out << "k"; break;
    case BuiltinType::UFract:       Out << "ur"; break;
    case BuiltinType::UAccum:       Out << "uk"; break;
    case BuiltinType::LongFract:    Out << "lr"; break;
    case BuiltinType::LongAccum:    Out << "lk"; break;
    case BuiltinType::ULongFract:   Out << "ulr"; break;
    case BuiltinType::ULongAccum:   Out << "ulk"; break;
  }
}

static void PrintFloatingLiteral(raw_ostream &Out, FloatingLiteral *Node,
                                 bool PrintSuffix) {
  SmallString<16> Str;
  Node->getValue().toString(Str);
  Out << Str;
  if (Str.find_first_not_of("-0123456789") == StringRef::npos)
    Out << '.'; // Trailing dot in order to separate from ints.

  if (!PrintSuffix)
    return;

  // Emit suffixes.  Float literals are always a builtin float type.
  switch (Node->getType()->castAs<BuiltinType>()->getKind()) {
  default: llvm_unreachable("Unexpected type for float literal!");
  case BuiltinType::Half:       break; // FIXME: suffix?
  case BuiltinType::Ibm128:     break; // FIXME: No suffix for ibm128 literal
  case BuiltinType::Double:     break; // no suffix.
  case BuiltinType::Float16:    Out << "F16"; break;
  case BuiltinType::Float:      Out << 'F'; break;
  case BuiltinType::LongDouble: Out << 'L'; break;
  case BuiltinType::Float128:   Out << 'Q'; break;
  }
}

void DeclPrinter::VisitFloatingLiteral(FloatingLiteral *Node) {
  PrintFloatingLiteral(Out, Node, /*PrintSuffix=*/true);
}

void DeclPrinter::VisitImaginaryLiteral(ImaginaryLiteral *Node) {
  PrintExpr(Node->getSubExpr());
  Out << "i";
}

void DeclPrinter::VisitStringLiteral(StringLiteral *Str) {
  Str->outputString(Out);
}

void DeclPrinter::VisitParenExpr(ParenExpr *Node) {
  Out << "(";
  PrintExpr(Node->getSubExpr());
  Out << ")";
}

void DeclPrinter::VisitUnaryOperator(UnaryOperator *Node) {
  if (!Node->isPostfix()) {
    Out << UnaryOperator::getOpcodeStr(Node->getOpcode());

    // Print a space if this is an "identifier operator" like __real, or if
    // it might be concatenated incorrectly like '+'.
    switch (Node->getOpcode()) {
    default: break;
    case UO_Real:
    case UO_Imag:
    case UO_Extension:
      Out << ' ';
      break;
    case UO_Plus:
    case UO_Minus:
      if (isa<UnaryOperator>(Node->getSubExpr()))
        Out << ' ';
      break;
    }
  }
  PrintExpr(Node->getSubExpr());

  if (Node->isPostfix())
    Out << UnaryOperator::getOpcodeStr(Node->getOpcode());
}

void DeclPrinter::VisitOffsetOfExpr(OffsetOfExpr *Node) {
  Out << "__builtin_offsetof(";
  Node->getTypeSourceInfo()->getType().print(Out, Policy);
  Out << ", ";
  bool PrintedSomething = false;
  for (unsigned i = 0, n = Node->getNumComponents(); i < n; ++i) {
    OffsetOfNode ON = Node->getComponent(i);
    if (ON.getKind() == OffsetOfNode::Array) {
      // Array node
      Out << "[";
      PrintExpr(Node->getIndexExpr(ON.getArrayExprIndex()));
      Out << "]";
      PrintedSomething = true;
      continue;
    }

    // Skip implicit base indirections.
    if (ON.getKind() == OffsetOfNode::Base)
      continue;

    // Field or identifier node.
    const IdentifierInfo *Id = ON.getFieldName();
    if (!Id)
      continue;

    if (PrintedSomething)
      Out << ".";
    else
      PrintedSomething = true;
    Out << Id->getName();
  }
  Out << ")";
}

void DeclPrinter::VisitUnaryExprOrTypeTraitExpr(
    UnaryExprOrTypeTraitExpr *Node) {
  const char *Spelling = getTraitSpelling(Node->getKind());
  if (Node->getKind() == UETT_AlignOf) {
    if (Policy.Alignof)
      Spelling = "alignof";
    else if (Policy.UnderscoreAlignof)
      Spelling = "_Alignof";
    else
      Spelling = "__alignof";
  }

  Out << Spelling;

  if (Node->isArgumentType()) {
    Out << '(';
    Node->getArgumentType().print(Out, Policy);
    Out << ')';
  } else {
    Out << " ";
    PrintExpr(Node->getArgumentExpr());
  }
}

void DeclPrinter::VisitGenericSelectionExpr(GenericSelectionExpr *Node) {
  Out << "_Generic(";
  if (Node->isExprPredicate())
    PrintExpr(Node->getControllingExpr());
  else
    Node->getControllingType()->getType().print(Out, Policy);

  for (const GenericSelectionExpr::Association &Assoc : Node->associations()) {
    Out << ", ";
    QualType T = Assoc.getType();
    if (T.isNull())
      Out << "default";
    else
      T.print(Out, Policy);
    Out << ": ";
    PrintExpr(Assoc.getAssociationExpr());
  }
  Out << ")";
}

void DeclPrinter::VisitArraySubscriptExpr(ArraySubscriptExpr *Node) {
  PrintExpr(Node->getLHS());
  Out << "[";
  PrintExpr(Node->getRHS());
  Out << "]";
}

void DeclPrinter::VisitMatrixSubscriptExpr(MatrixSubscriptExpr *Node) {
  PrintExpr(Node->getBase());
  Out << "[";
  PrintExpr(Node->getRowIdx());
  Out << "]";
  Out << "[";
  PrintExpr(Node->getColumnIdx());
  Out << "]";
}

void DeclPrinter::VisitArraySectionExpr(ArraySectionExpr *Node) {
  PrintExpr(Node->getBase());
  Out << "[";
  if (Node->getLowerBound())
    PrintExpr(Node->getLowerBound());
  if (Node->getColonLocFirst().isValid()) {
    Out << ":";
    if (Node->getLength())
      PrintExpr(Node->getLength());
  }
  if (Node->isOMPArraySection() && Node->getColonLocSecond().isValid()) {
    Out << ":";
    if (Node->getStride())
      PrintExpr(Node->getStride());
  }
  Out << "]";
}

void DeclPrinter::VisitOMPArrayShapingExpr(OMPArrayShapingExpr *Node) {
  Out << "(";
  for (Expr *E : Node->getDimensions()) {
    Out << "[";
    PrintExpr(E);
    Out << "]";
  }
  Out << ")";
  PrintExpr(Node->getBase());
}

void DeclPrinter::VisitOMPIteratorExpr(OMPIteratorExpr *Node) {
  Out << "iterator(";
  for (unsigned I = 0, E = Node->numOfIterators(); I < E; ++I) {
    auto *VD = cast<ValueDecl>(Node->getIteratorDecl(I));
    VD->getType().print(Out, Policy);
    const OMPIteratorExpr::IteratorRange Range = Node->getIteratorRange(I);
    Out << " " << VD->getName() << " = ";
    PrintExpr(Range.Begin);
    Out << ":";
    PrintExpr(Range.End);
    if (Range.Step) {
      Out << ":";
      PrintExpr(Range.Step);
    }
    if (I < E - 1)
      Out << ", ";
  }
  Out << ")";
}

void DeclPrinter::PrintCallArgs(CallExpr *Call) {
  for (unsigned i = 0, e = Call->getNumArgs(); i != e; ++i) {
    if (isa<CXXDefaultArgExpr>(Call->getArg(i))) {
      // Don't print any defaulted arguments
      break;
    }

    if (i) Out << ", ";
    PrintExpr(Call->getArg(i));
  }
}

void DeclPrinter::VisitCallExpr(CallExpr *Call) {
  PrintExpr(Call->getCallee());
  Out << "(";
  PrintCallArgs(Call);
  Out << ")";
}

bool DeclPrinter::isImplicitThis(const Expr *E) {
  if (const auto *TE = dyn_cast<CXXThisExpr>(E))
    return TE->isImplicit();
  return false;
}

void DeclPrinter::VisitMemberExpr(MemberExpr *Node) {
  if (!Policy.SuppressImplicitBase || !isImplicitThis(Node->getBase())) {
    PrintExpr(Node->getBase());

    auto *ParentMember = dyn_cast<MemberExpr>(Node->getBase());
    FieldDecl *ParentDecl =
        ParentMember ? dyn_cast<FieldDecl>(ParentMember->getMemberDecl())
                     : nullptr;

    if (!ParentDecl || !ParentDecl->isAnonymousStructOrUnion())
      Out << (Node->isArrow() ? "->" : ".");
  }

  if (auto *FD = dyn_cast<FieldDecl>(Node->getMemberDecl()))
    if (FD->isAnonymousStructOrUnion())
      return;

  if (NestedNameSpecifier *Qualifier = Node->getQualifier())
    Qualifier->print(Out, Policy);
  if (Node->hasTemplateKeyword())
    Out << "template ";
  Out << Node->getMemberNameInfo();
  const TemplateParameterList *TPL = nullptr;
  if (auto *FD = dyn_cast<FunctionDecl>(Node->getMemberDecl())) {
    if (!Node->hadMultipleCandidates())
      if (auto *FTD = FD->getPrimaryTemplate())
        TPL = FTD->getTemplateParameters();
  } else if (auto *VTSD =
                 dyn_cast<VarTemplateSpecializationDecl>(Node->getMemberDecl()))
    TPL = VTSD->getSpecializedTemplate()->getTemplateParameters();
  if (Node->hasExplicitTemplateArgs())
    printTemplateArgumentList(Out, Node->template_arguments(), Policy, TPL);
}

void DeclPrinter::VisitObjCIsaExpr(ObjCIsaExpr *Node) {
  PrintExpr(Node->getBase());
  Out << (Node->isArrow() ? "->isa" : ".isa");
}

void DeclPrinter::VisitExtVectorElementExpr(ExtVectorElementExpr *Node) {
  PrintExpr(Node->getBase());
  Out << ".";
  Out << Node->getAccessor().getName();
}

void DeclPrinter::VisitCStyleCastExpr(CStyleCastExpr *Node) {
  Out << '(';
  Node->getTypeAsWritten().print(Out, Policy);
  Out << ')';
  PrintExpr(Node->getSubExpr());
}

void DeclPrinter::VisitCompoundLiteralExpr(CompoundLiteralExpr *Node) {
  Out << '(';
  Node->getType().print(Out, Policy);
  Out << ')';
  PrintExpr(Node->getInitializer());
}

void DeclPrinter::VisitImplicitCastExpr(ImplicitCastExpr *Node) {
  // No need to print anything, simply forward to the subexpression.
  PrintExpr(Node->getSubExpr());
}

void DeclPrinter::VisitBinaryOperator(BinaryOperator *Node) {
/****************************************************************************/
  if (Node->getOpcode() == BO_LMG) {
    Out << "@Expression";
    std::vector<std::vector<int>> shapes(calc->getNumParams(),
                                         std::vector<int>());
    for (int paramCount = 0; paramCount < calc->getNumParams(); paramCount++) {
      dacppTranslator::Param *param = calc->getParam(paramCount);
      for (int dimCount = 0; dimCount < param->getDim(); dimCount++) {
        // shapes[paramCount].push_back(param->getShape(dimCount));
      }
    }
    calc->setExpr(Node, shapes);
    return;
  }
/****************************************************************************/
  PrintExpr(Node->getLHS());
  Out << " " << BinaryOperator::getOpcodeStr(Node->getOpcode()) << " ";
  PrintExpr(Node->getRHS());
}

void DeclPrinter::VisitCompoundAssignOperator(CompoundAssignOperator *Node) {
  PrintExpr(Node->getLHS());
  Out << " " << BinaryOperator::getOpcodeStr(Node->getOpcode()) << " ";
  PrintExpr(Node->getRHS());
}

void DeclPrinter::VisitConditionalOperator(ConditionalOperator *Node) {
  PrintExpr(Node->getCond());
  Out << " ? ";
  PrintExpr(Node->getLHS());
  Out << " : ";
  PrintExpr(Node->getRHS());
}

// GNU extensions.

void
DeclPrinter::VisitBinaryConditionalOperator(BinaryConditionalOperator *Node) {
  PrintExpr(Node->getCommon());
  Out << " ?: ";
  PrintExpr(Node->getFalseExpr());
}

void DeclPrinter::VisitAddrLabelExpr(AddrLabelExpr *Node) {
  Out << "&&" << Node->getLabel()->getName();
}

void DeclPrinter::VisitStmtExpr(StmtExpr *E) {
  Out << "(";
  PrintRawCompoundStmt(E->getSubStmt());
  Out << ")";
}

void DeclPrinter::VisitChooseExpr(ChooseExpr *Node) {
  Out << "__builtin_choose_expr(";
  PrintExpr(Node->getCond());
  Out << ", ";
  PrintExpr(Node->getLHS());
  Out << ", ";
  PrintExpr(Node->getRHS());
  Out << ")";
}

void DeclPrinter::VisitGNUNullExpr(GNUNullExpr *) {
  Out << "__null";
}

void DeclPrinter::VisitShuffleVectorExpr(ShuffleVectorExpr *Node) {
  Out << "__builtin_shufflevector(";
  for (unsigned i = 0, e = Node->getNumSubExprs(); i != e; ++i) {
    if (i) Out << ", ";
    PrintExpr(Node->getExpr(i));
  }
  Out << ")";
}

void DeclPrinter::VisitConvertVectorExpr(ConvertVectorExpr *Node) {
  Out << "__builtin_convertvector(";
  PrintExpr(Node->getSrcExpr());
  Out << ", ";
  Node->getType().print(Out, Policy);
  Out << ")";
}

void DeclPrinter::VisitInitListExpr(InitListExpr* Node) {
  if (Node->getSyntacticForm()) {
    Visit(Node->getSyntacticForm());
    return;
  }

  Out << "{";
  for (unsigned i = 0, e = Node->getNumInits(); i != e; ++i) {
    if (i) Out << ", ";
    if (Node->getInit(i))
      PrintExpr(Node->getInit(i));
    else
      Out << "{}";
  }
  Out << "}";
}

void DeclPrinter::VisitArrayInitLoopExpr(ArrayInitLoopExpr *Node) {
  // There's no way to express this expression in any of our supported
  // languages, so just emit something terse and (hopefully) clear.
  Out << "{";
  PrintExpr(Node->getSubExpr());
  Out << "}";
}

void DeclPrinter::VisitArrayInitIndexExpr(ArrayInitIndexExpr *Node) {
  Out << "*";
}

void DeclPrinter::VisitParenListExpr(ParenListExpr* Node) {
  Out << "(";
  for (unsigned i = 0, e = Node->getNumExprs(); i != e; ++i) {
    if (i) Out << ", ";
    PrintExpr(Node->getExpr(i));
  }
  Out << ")";
}

void DeclPrinter::VisitDesignatedInitExpr(DesignatedInitExpr *Node) {
  bool NeedsEquals = true;
  for (const DesignatedInitExpr::Designator &D : Node->designators()) {
    if (D.isFieldDesignator()) {
      if (D.getDotLoc().isInvalid()) {
        if (const IdentifierInfo *II = D.getFieldName()) {
          Out << II->getName() << ":";
          NeedsEquals = false;
        }
      } else {
        Out << "." << D.getFieldName()->getName();
      }
    } else {
      Out << "[";
      if (D.isArrayDesignator()) {
        PrintExpr(Node->getArrayIndex(D));
      } else {
        PrintExpr(Node->getArrayRangeStart(D));
        Out << " ... ";
        PrintExpr(Node->getArrayRangeEnd(D));
      }
      Out << "]";
    }
  }

  if (NeedsEquals)
    Out << " = ";
  else
    Out << " ";
  PrintExpr(Node->getInit());
}

void DeclPrinter::VisitDesignatedInitUpdateExpr(
    DesignatedInitUpdateExpr *Node) {
  Out << "{";
  Out << "/*base*/";
  PrintExpr(Node->getBase());
  Out << ", ";

  Out << "/*updater*/";
  PrintExpr(Node->getUpdater());
  Out << "}";
}

void DeclPrinter::VisitNoInitExpr(NoInitExpr *Node) {
  Out << "/*no init*/";
}

void DeclPrinter::VisitImplicitValueInitExpr(ImplicitValueInitExpr *Node) {
  if (Node->getType()->getAsCXXRecordDecl()) {
    Out << "/*implicit*/";
    Node->getType().print(Out, Policy);
    Out << "()";
  } else {
    Out << "/*implicit*/(";
    Node->getType().print(Out, Policy);
    Out << ')';
    if (Node->getType()->isRecordType())
      Out << "{}";
    else
      Out << 0;
  }
}

void DeclPrinter::VisitVAArgExpr(VAArgExpr *Node) {
  Out << "__builtin_va_arg(";
  PrintExpr(Node->getSubExpr());
  Out << ", ";
  Node->getType().print(Out, Policy);
  Out << ")";
}

void DeclPrinter::VisitPseudoObjectExpr(PseudoObjectExpr *Node) {
  PrintExpr(Node->getSyntacticForm());
}

void DeclPrinter::VisitAtomicExpr(AtomicExpr *Node) {
  const char *Name = nullptr;
  switch (Node->getOp()) {
#define BUILTIN(ID, TYPE, ATTRS)
#define ATOMIC_BUILTIN(ID, TYPE, ATTRS) \
  case AtomicExpr::AO ## ID: \
    Name = #ID "("; \
    break;
#include "clang/Basic/Builtins.inc"
  }
  Out << Name;

  // AtomicExpr stores its subexpressions in a permuted order.
  PrintExpr(Node->getPtr());
  if (Node->getOp() != AtomicExpr::AO__c11_atomic_load &&
      Node->getOp() != AtomicExpr::AO__atomic_load_n &&
      Node->getOp() != AtomicExpr::AO__scoped_atomic_load_n &&
      Node->getOp() != AtomicExpr::AO__opencl_atomic_load &&
      Node->getOp() != AtomicExpr::AO__hip_atomic_load) {
    Out << ", ";
    PrintExpr(Node->getVal1());
  }
  if (Node->getOp() == AtomicExpr::AO__atomic_exchange ||
      Node->isCmpXChg()) {
    Out << ", ";
    PrintExpr(Node->getVal2());
  }
  if (Node->getOp() == AtomicExpr::AO__atomic_compare_exchange ||
      Node->getOp() == AtomicExpr::AO__atomic_compare_exchange_n) {
    Out << ", ";
    PrintExpr(Node->getWeak());
  }
  if (Node->getOp() != AtomicExpr::AO__c11_atomic_init &&
      Node->getOp() != AtomicExpr::AO__opencl_atomic_init) {
    Out << ", ";
    PrintExpr(Node->getOrder());
  }
  if (Node->isCmpXChg()) {
    Out << ", ";
    PrintExpr(Node->getOrderFail());
  }
  Out << ")";
}

// C++
void DeclPrinter::VisitCXXOperatorCallExpr(CXXOperatorCallExpr *Node) {
  OverloadedOperatorKind Kind = Node->getOperator();
  if (Kind == OO_PlusPlus || Kind == OO_MinusMinus) {
    if (Node->getNumArgs() == 1) {
      Out << getOperatorSpelling(Kind) << ' ';
      PrintExpr(Node->getArg(0));
    } else {
      PrintExpr(Node->getArg(0));
      Out << ' ' << getOperatorSpelling(Kind);
    }
  } else if (Kind == OO_Arrow) {
    PrintExpr(Node->getArg(0));
  } else if (Kind == OO_Call || Kind == OO_Subscript) {
    PrintExpr(Node->getArg(0));
    Out << (Kind == OO_Call ? '(' : '[');
    for (unsigned ArgIdx = 1; ArgIdx < Node->getNumArgs(); ++ArgIdx) {
      if (ArgIdx > 1)
        Out << ", ";
      if (!isa<CXXDefaultArgExpr>(Node->getArg(ArgIdx)))
        PrintExpr(Node->getArg(ArgIdx));
    }
    Out << (Kind == OO_Call ? ')' : ']');
  } else if (Node->getNumArgs() == 1) {
    Out << getOperatorSpelling(Kind) << ' ';
    PrintExpr(Node->getArg(0));
  } else if (Node->getNumArgs() == 2) {
    PrintExpr(Node->getArg(0));
    Out << ' ' << getOperatorSpelling(Kind) << ' ';
    PrintExpr(Node->getArg(1));
  } else {
    llvm_unreachable("unknown overloaded operator");
  }
}

void DeclPrinter::VisitCXXMemberCallExpr(CXXMemberCallExpr *Node) {
/****************************************************************************/
std::string Msg;
llvm::raw_string_ostream SS(Msg);
if (const auto *Call = dyn_cast<CallExpr>(Node)) {
  if (auto *Callee = Call->getCallee()) {
    if (auto *ME = dyn_cast<MemberExpr>(Callee)) {
      if (!isImplicitThis(ME->getBase())) {
        if (const Expr *Base = ME->getBase()) {
          if (auto *DRE = dyn_cast<DeclRefExpr>(Base)) {
            if (strstr(DRE->getDecl()->getType().getAsString().c_str(),
                       "Tensor") &&
                !strcmp("getShape",
                        ME->getMemberNameInfo().getAsString().c_str())) {
              for (int i = 0; i < calc->getNumParams(); i++) {
                if (!strcmp(DRE->getNameInfo().getAsString().c_str(),
                            calc->getParam(i)->getName().c_str())) {
                  Call->getArg(0)->printPretty(SS, Helper, Policy);
                  Out << calc->getParam(i)->getShape(atoi(SS.str().c_str()));
                  break;
                }
              }
              return;
            }
          }
        }
      }
    }
  }
}
/****************************************************************************/

// If we have a conversion operator call only print the argument.
CXXMethodDecl *MD = Node->getMethodDecl();
if (isa_and_nonnull<CXXConversionDecl>(MD)) {
  PrintExpr(Node->getImplicitObjectArgument());
  return;
  }
  VisitCallExpr(cast<CallExpr>(Node));
}

void DeclPrinter::VisitCUDAKernelCallExpr(CUDAKernelCallExpr *Node) {
  PrintExpr(Node->getCallee());
  Out << "<<<";
  PrintCallArgs(Node->getConfig());
  Out << ">>>(";
  PrintCallArgs(Node);
  Out << ")";
}

void DeclPrinter::VisitCXXRewrittenBinaryOperator(
    CXXRewrittenBinaryOperator *Node) {
  CXXRewrittenBinaryOperator::DecomposedForm Decomposed =
      Node->getDecomposedForm();
  PrintExpr(const_cast<Expr*>(Decomposed.LHS));
  Out << ' ' << BinaryOperator::getOpcodeStr(Decomposed.Opcode) << ' ';
  PrintExpr(const_cast<Expr*>(Decomposed.RHS));
}

void DeclPrinter::VisitCXXNamedCastExpr(CXXNamedCastExpr *Node) {
  Out << Node->getCastName() << '<';
  Node->getTypeAsWritten().print(Out, Policy);
  Out << ">(";
  PrintExpr(Node->getSubExpr());
  Out << ")";
}

void DeclPrinter::VisitCXXStaticCastExpr(CXXStaticCastExpr *Node) {
  VisitCXXNamedCastExpr(Node);
}

void DeclPrinter::VisitCXXDynamicCastExpr(CXXDynamicCastExpr *Node) {
  VisitCXXNamedCastExpr(Node);
}

void DeclPrinter::VisitCXXReinterpretCastExpr(CXXReinterpretCastExpr *Node) {
  VisitCXXNamedCastExpr(Node);
}

void DeclPrinter::VisitCXXConstCastExpr(CXXConstCastExpr *Node) {
  VisitCXXNamedCastExpr(Node);
}

void DeclPrinter::VisitBuiltinBitCastExpr(BuiltinBitCastExpr *Node) {
  Out << "__builtin_bit_cast(";
  Node->getTypeInfoAsWritten()->getType().print(Out, Policy);
  Out << ", ";
  PrintExpr(Node->getSubExpr());
  Out << ")";
}

void DeclPrinter::VisitCXXAddrspaceCastExpr(CXXAddrspaceCastExpr *Node) {
  VisitCXXNamedCastExpr(Node);
}

void DeclPrinter::VisitCXXTypeidExpr(CXXTypeidExpr *Node) {
  Out << "typeid(";
  if (Node->isTypeOperand()) {
    Node->getTypeOperandSourceInfo()->getType().print(Out, Policy);
  } else {
    PrintExpr(Node->getExprOperand());
  }
  Out << ")";
}

void DeclPrinter::VisitCXXUuidofExpr(CXXUuidofExpr *Node) {
  Out << "__uuidof(";
  if (Node->isTypeOperand()) {
    Node->getTypeOperandSourceInfo()->getType().print(Out, Policy);
  } else {
    PrintExpr(Node->getExprOperand());
  }
  Out << ")";
}

void DeclPrinter::VisitMSPropertyRefExpr(MSPropertyRefExpr *Node) {
  PrintExpr(Node->getBaseExpr());
  if (Node->isArrow())
    Out << "->";
  else
    Out << ".";
  if (NestedNameSpecifier *Qualifier =
      Node->getQualifierLoc().getNestedNameSpecifier())
    Qualifier->print(Out, Policy);
  Out << Node->getPropertyDecl()->getDeclName();
}

void DeclPrinter::VisitMSPropertySubscriptExpr(MSPropertySubscriptExpr *Node) {
  PrintExpr(Node->getBase());
  Out << "[";
  PrintExpr(Node->getIdx());
  Out << "]";
}

void DeclPrinter::VisitUserDefinedLiteral(UserDefinedLiteral *Node) {
  switch (Node->getLiteralOperatorKind()) {
  case UserDefinedLiteral::LOK_Raw:
    Out << cast<StringLiteral>(Node->getArg(0)->IgnoreImpCasts())->getString();
    break;
  case UserDefinedLiteral::LOK_Template: {
    const auto *DRE = cast<DeclRefExpr>(Node->getCallee()->IgnoreImpCasts());
    const TemplateArgumentList *Args =
      cast<FunctionDecl>(DRE->getDecl())->getTemplateSpecializationArgs();
    assert(Args);

    if (Args->size() != 1 || Args->get(0).getKind() != TemplateArgument::Pack) {
      const TemplateParameterList *TPL = nullptr;
      if (!DRE->hadMultipleCandidates())
        if (const auto *TD = dyn_cast<TemplateDecl>(DRE->getDecl()))
          TPL = TD->getTemplateParameters();
      Out << "operator\"\"" << Node->getUDSuffix()->getName();
      printTemplateArgumentList(Out, Args->asArray(), Policy, TPL);
      Out << "()";
      return;
    }

    const TemplateArgument &Pack = Args->get(0);
    for (const auto &P : Pack.pack_elements()) {
      char C = (char)P.getAsIntegral().getZExtValue();
      Out << C;
    }
    break;
  }
  case UserDefinedLiteral::LOK_Integer: {
    // Print integer literal without suffix.
    const auto *Int = cast<IntegerLiteral>(Node->getCookedLiteral());
    Out << toString(Int->getValue(), 10, /*isSigned*/false);
    break;
  }
  case UserDefinedLiteral::LOK_Floating: {
    // Print floating literal without suffix.
    auto *Float = cast<FloatingLiteral>(Node->getCookedLiteral());
    PrintFloatingLiteral(Out, Float, /*PrintSuffix=*/false);
    break;
  }
  case UserDefinedLiteral::LOK_String:
  case UserDefinedLiteral::LOK_Character:
    PrintExpr(Node->getCookedLiteral());
    break;
  }
  Out << Node->getUDSuffix()->getName();
}

void DeclPrinter::VisitCXXBoolLiteralExpr(CXXBoolLiteralExpr *Node) {
  Out << (Node->getValue() ? "true" : "false");
}

void DeclPrinter::VisitCXXNullPtrLiteralExpr(CXXNullPtrLiteralExpr *Node) {
  Out << "nullptr";
}

void DeclPrinter::VisitCXXThisExpr(CXXThisExpr *Node) {
  Out << "this";
}

void DeclPrinter::VisitCXXThrowExpr(CXXThrowExpr *Node) {
  if (!Node->getSubExpr())
    Out << "throw";
  else {
    Out << "throw ";
    PrintExpr(Node->getSubExpr());
  }
}

void DeclPrinter::VisitCXXDefaultArgExpr(CXXDefaultArgExpr *Node) {
  // Nothing to print: we picked up the default argument.
}

void DeclPrinter::VisitCXXDefaultInitExpr(CXXDefaultInitExpr *Node) {
  // Nothing to print: we picked up the default initializer.
}

void DeclPrinter::VisitCXXFunctionalCastExpr(CXXFunctionalCastExpr *Node) {
  auto TargetType = Node->getType();
  auto *Auto = TargetType->getContainedDeducedType();
  bool Bare = Auto && Auto->isDeduced();

  // Parenthesize deduced casts.
  if (Bare)
    Out << '(';
  TargetType.print(Out, Policy);
  if (Bare)
    Out << ')';

  // No extra braces surrounding the inner construct.
  if (!Node->isListInitialization())
    Out << '(';
  PrintExpr(Node->getSubExpr());
  if (!Node->isListInitialization())
    Out << ')';
}

void DeclPrinter::VisitCXXBindTemporaryExpr(CXXBindTemporaryExpr *Node) {
  PrintExpr(Node->getSubExpr());
}

void DeclPrinter::VisitCXXTemporaryObjectExpr(CXXTemporaryObjectExpr *Node) {
  Node->getType().print(Out, Policy);
  if (Node->isStdInitListInitialization())
    /* Nothing to do; braces are part of creating the std::initializer_list. */;
  else if (Node->isListInitialization())
    Out << "{";
  else
    Out << "(";
  for (CXXTemporaryObjectExpr::arg_iterator Arg = Node->arg_begin(),
                                         ArgEnd = Node->arg_end();
       Arg != ArgEnd; ++Arg) {
    if ((*Arg)->isDefaultArgument())
      break;
    if (Arg != Node->arg_begin())
      Out << ", ";
    PrintExpr(*Arg);
  }
  if (Node->isStdInitListInitialization())
    /* See above. */;
  else if (Node->isListInitialization())
    Out << "}";
  else
    Out << ")";
}

void DeclPrinter::VisitLambdaExpr(LambdaExpr *Node) {
  Out << '[';
  bool NeedComma = false;
  switch (Node->getCaptureDefault()) {
  case LCD_None:
    break;

  case LCD_ByCopy:
    Out << '=';
    NeedComma = true;
    break;

  case LCD_ByRef:
    Out << '&';
    NeedComma = true;
    break;
  }
  for (LambdaExpr::capture_iterator C = Node->explicit_capture_begin(),
                                 CEnd = Node->explicit_capture_end();
       C != CEnd;
       ++C) {
    if (C->capturesVLAType())
      continue;

    if (NeedComma)
      Out << ", ";
    NeedComma = true;

    switch (C->getCaptureKind()) {
    case LCK_This:
      Out << "this";
      break;

    case LCK_StarThis:
      Out << "*this";
      break;

    case LCK_ByRef:
      if (Node->getCaptureDefault() != LCD_ByRef || Node->isInitCapture(C))
        Out << '&';
      Out << C->getCapturedVar()->getName();
      break;

    case LCK_ByCopy:
      Out << C->getCapturedVar()->getName();
      break;

    case LCK_VLAType:
      llvm_unreachable("VLA type in explicit captures.");
    }

    if (C->isPackExpansion())
      Out << "...";

    if (Node->isInitCapture(C)) {
      // Init captures are always VarDecl.
      auto *D = cast<VarDecl>(C->getCapturedVar());

      llvm::StringRef Pre;
      llvm::StringRef Post;
      if (D->getInitStyle() == VarDecl::CallInit &&
          !isa<ParenListExpr>(D->getInit())) {
        Pre = "(";
        Post = ")";
      } else if (D->getInitStyle() == VarDecl::CInit) {
        Pre = " = ";
      }

      Out << Pre;
      PrintExpr(D->getInit());
      Out << Post;
    }
  }
  Out << ']';

  if (!Node->getExplicitTemplateParameters().empty()) {
    Node->getTemplateParameterList()->print(
        Out, Node->getLambdaClass()->getASTContext(),
        /*OmitTemplateKW*/true);
  }

  if (Node->hasExplicitParameters()) {
    Out << '(';
    CXXMethodDecl *Method = Node->getCallOperator();
    NeedComma = false;
    for (const auto *P : Method->parameters()) {
      if (NeedComma) {
        Out << ", ";
      } else {
        NeedComma = true;
      }
      std::string ParamStr =
          (Policy.CleanUglifiedParameters && P->getIdentifier())
              ? P->getIdentifier()->deuglifiedName().str()
              : P->getNameAsString();
      P->getOriginalType().print(Out, Policy, ParamStr);
    }
    if (Method->isVariadic()) {
      if (NeedComma)
        Out << ", ";
      Out << "...";
    }
    Out << ')';

    if (Node->isMutable())
      Out << " mutable";

    auto *Proto = Method->getType()->castAs<FunctionProtoType>();
    Proto->printExceptionSpecification(Out, Policy);

    // FIXME: Attributes

    // Print the trailing return type if it was specified in the source.
    if (Node->hasExplicitResultType()) {
      Out << " -> ";
      Proto->getReturnType().print(Out, Policy);
    }
  }

  // Print the body.
  Out << ' ';
  if (Policy.TerseOutput)
    Out << "{}";
  else
    PrintRawCompoundStmt(Node->getCompoundStmtBody());
}

void DeclPrinter::VisitCXXScalarValueInitExpr(CXXScalarValueInitExpr *Node) {
  if (TypeSourceInfo *TSInfo = Node->getTypeSourceInfo())
    TSInfo->getType().print(Out, Policy);
  else
    Node->getType().print(Out, Policy);
  Out << "()";
}

void DeclPrinter::VisitCXXNewExpr(CXXNewExpr *E) {
  if (E->isGlobalNew())
    Out << "::";
  Out << "new ";
  unsigned NumPlace = E->getNumPlacementArgs();
  if (NumPlace > 0 && !isa<CXXDefaultArgExpr>(E->getPlacementArg(0))) {
    Out << "(";
    PrintExpr(E->getPlacementArg(0));
    for (unsigned i = 1; i < NumPlace; ++i) {
      if (isa<CXXDefaultArgExpr>(E->getPlacementArg(i)))
        break;
      Out << ", ";
      PrintExpr(E->getPlacementArg(i));
    }
    Out << ") ";
  }
  if (E->isParenTypeId())
    Out << "(";
  std::string TypeS;
  if (E->isArray()) {
    llvm::raw_string_ostream s(TypeS);
    s << '[';
    if (std::optional<Expr *> Size = E->getArraySize())
      printPretty(*Size, s, Helper, Policy, 0, "\n", nullptr);
    s << ']';
  }
  E->getAllocatedType().print(Out, Policy, TypeS);
  if (E->isParenTypeId())
    Out << ")";

  CXXNewInitializationStyle InitStyle = E->getInitializationStyle();
  if (InitStyle != CXXNewInitializationStyle::None) {
    bool Bare = InitStyle == CXXNewInitializationStyle::Parens &&
                !isa<ParenListExpr>(E->getInitializer());
    if (Bare)
      Out << "(";
    PrintExpr(E->getInitializer());
    if (Bare)
      Out << ")";
  }
}

void DeclPrinter::VisitCXXDeleteExpr(CXXDeleteExpr *E) {
  if (E->isGlobalDelete())
    Out << "::";
  Out << "delete ";
  if (E->isArrayForm())
    Out << "[] ";
  PrintExpr(E->getArgument());
}

void DeclPrinter::VisitCXXPseudoDestructorExpr(CXXPseudoDestructorExpr *E) {
  PrintExpr(E->getBase());
  if (E->isArrow())
    Out << "->";
  else
    Out << '.';
  if (E->getQualifier())
    E->getQualifier()->print(Out, Policy);
  Out << "~";

  if (const IdentifierInfo *II = E->getDestroyedTypeIdentifier())
    Out << II->getName();
  else
    E->getDestroyedType().print(Out, Policy);
}

void DeclPrinter::VisitCXXConstructExpr(CXXConstructExpr *E) {
  if (E->isListInitialization() && !E->isStdInitListInitialization())
    Out << "{";

  for (unsigned i = 0, e = E->getNumArgs(); i != e; ++i) {
    if (isa<CXXDefaultArgExpr>(E->getArg(i))) {
      // Don't print any defaulted arguments
      break;
    }

    if (i) Out << ", ";
    PrintExpr(E->getArg(i));
  }

  if (E->isListInitialization() && !E->isStdInitListInitialization())
    Out << "}";
}

void DeclPrinter::VisitCXXInheritedCtorInitExpr(CXXInheritedCtorInitExpr *E) {
  // Parens are printed by the surrounding context.
  Out << "<forwarded>";
}

void DeclPrinter::VisitCXXStdInitializerListExpr(CXXStdInitializerListExpr *E) {
  PrintExpr(E->getSubExpr());
}

void DeclPrinter::VisitExprWithCleanups(ExprWithCleanups *E) {
  // Just forward to the subexpression.
  PrintExpr(E->getSubExpr());
}

void DeclPrinter::VisitCXXUnresolvedConstructExpr(
    CXXUnresolvedConstructExpr *Node) {
  Node->getTypeAsWritten().print(Out, Policy);
  if (!Node->isListInitialization())
    Out << '(';
  for (auto Arg = Node->arg_begin(), ArgEnd = Node->arg_end(); Arg != ArgEnd;
       ++Arg) {
    if (Arg != Node->arg_begin())
      Out << ", ";
    PrintExpr(*Arg);
  }
  if (!Node->isListInitialization())
    Out << ')';
}

void DeclPrinter::VisitCXXDependentScopeMemberExpr(
                                         CXXDependentScopeMemberExpr *Node) {
  if (!Node->isImplicitAccess()) {
    PrintExpr(Node->getBase());
    Out << (Node->isArrow() ? "->" : ".");
  }
  if (NestedNameSpecifier *Qualifier = Node->getQualifier())
    Qualifier->print(Out, Policy);
  if (Node->hasTemplateKeyword())
    Out << "template ";
  Out << Node->getMemberNameInfo();
  if (Node->hasExplicitTemplateArgs())
    printTemplateArgumentList(Out, Node->template_arguments(), Policy);
}

void DeclPrinter::VisitUnresolvedMemberExpr(UnresolvedMemberExpr *Node) {
  if (!Node->isImplicitAccess()) {
    PrintExpr(Node->getBase());
    Out << (Node->isArrow() ? "->" : ".");
  }
  if (NestedNameSpecifier *Qualifier = Node->getQualifier())
    Qualifier->print(Out, Policy);
  if (Node->hasTemplateKeyword())
    Out << "template ";
  Out << Node->getMemberNameInfo();
  if (Node->hasExplicitTemplateArgs())
    printTemplateArgumentList(Out, Node->template_arguments(), Policy);
}

void DeclPrinter::VisitTypeTraitExpr(TypeTraitExpr *E) {
  Out << getTraitSpelling(E->getTrait()) << "(";
  for (unsigned I = 0, N = E->getNumArgs(); I != N; ++I) {
    if (I > 0)
      Out << ", ";
    E->getArg(I)->getType().print(Out, Policy);
  }
  Out << ")";
}

void DeclPrinter::VisitArrayTypeTraitExpr(ArrayTypeTraitExpr *E) {
  Out << getTraitSpelling(E->getTrait()) << '(';
  E->getQueriedType().print(Out, Policy);
  Out << ')';
}

void DeclPrinter::VisitExpressionTraitExpr(ExpressionTraitExpr *E) {
  Out << getTraitSpelling(E->getTrait()) << '(';
  PrintExpr(E->getQueriedExpression());
  Out << ')';
}

void DeclPrinter::VisitCXXNoexceptExpr(CXXNoexceptExpr *E) {
  Out << "noexcept(";
  PrintExpr(E->getOperand());
  Out << ")";
}

void DeclPrinter::VisitPackExpansionExpr(PackExpansionExpr *E) {
  PrintExpr(E->getPattern());
  Out << "...";
}

void DeclPrinter::VisitSizeOfPackExpr(SizeOfPackExpr *E) {
  Out << "sizeof...(" << *E->getPack() << ")";
}

void DeclPrinter::VisitPackIndexingExpr(PackIndexingExpr *E) {
  Out << E->getPackIdExpression() << "...[" << E->getIndexExpr() << "]";
}

void DeclPrinter::VisitSubstNonTypeTemplateParmPackExpr(
                                       SubstNonTypeTemplateParmPackExpr *Node) {
  Out << *Node->getParameterPack();
}

void DeclPrinter::VisitSubstNonTypeTemplateParmExpr(
                                       SubstNonTypeTemplateParmExpr *Node) {
  Visit(Node->getReplacement());
}

void DeclPrinter::VisitFunctionParmPackExpr(FunctionParmPackExpr *E) {
  Out << *E->getParameterPack();
}

void DeclPrinter::VisitMaterializeTemporaryExpr(MaterializeTemporaryExpr *Node){
  PrintExpr(Node->getSubExpr());
}

void DeclPrinter::VisitCXXFoldExpr(CXXFoldExpr *E) {
  Out << "(";
  if (E->getLHS()) {
    PrintExpr(E->getLHS());
    Out << " " << BinaryOperator::getOpcodeStr(E->getOperator()) << " ";
  }
  Out << "...";
  if (E->getRHS()) {
    Out << " " << BinaryOperator::getOpcodeStr(E->getOperator()) << " ";
    PrintExpr(E->getRHS());
  }
  Out << ")";
}

void DeclPrinter::VisitCXXParenListInitExpr(CXXParenListInitExpr *Node) {
  Out << "(";
  llvm::interleaveComma(Node->getInitExprs(), Out,
                        [&](Expr *E) { PrintExpr(E); });
  Out << ")";
}

void DeclPrinter::VisitConceptSpecializationExpr(ConceptSpecializationExpr *E) {
  NestedNameSpecifierLoc NNS = E->getNestedNameSpecifierLoc();
  if (NNS)
    NNS.getNestedNameSpecifier()->print(Out, Policy);
  if (E->getTemplateKWLoc().isValid())
    Out << "template ";
  Out << E->getFoundDecl()->getName();
  printTemplateArgumentList(Out, E->getTemplateArgsAsWritten()->arguments(),
                            Policy,
                            E->getNamedConcept()->getTemplateParameters());
}

void DeclPrinter::VisitRequiresExpr(RequiresExpr *E) {
  Out << "requires ";
  auto LocalParameters = E->getLocalParameters();
  if (!LocalParameters.empty()) {
    Out << "(";
    for (ParmVarDecl *LocalParam : LocalParameters) {
      PrintRawDecl(LocalParam);
      if (LocalParam != LocalParameters.back())
        Out << ", ";
    }

    Out << ") ";
  }
  Out << "{ ";
  auto Requirements = E->getRequirements();
  for (concepts::Requirement *Req : Requirements) {
    if (auto *TypeReq = dyn_cast<concepts::TypeRequirement>(Req)) {
      if (TypeReq->isSubstitutionFailure())
        Out << "<<error-type>>";
      else
        TypeReq->getType()->getType().print(Out, Policy);
    } else if (auto *ExprReq = dyn_cast<concepts::ExprRequirement>(Req)) {
      if (ExprReq->isCompound())
        Out << "{ ";
      if (ExprReq->isExprSubstitutionFailure())
        Out << "<<error-expression>>";
      else
        PrintExpr(ExprReq->getExpr());
      if (ExprReq->isCompound()) {
        Out << " }";
        if (ExprReq->getNoexceptLoc().isValid())
          Out << " noexcept";
        const auto &RetReq = ExprReq->getReturnTypeRequirement();
        if (!RetReq.isEmpty()) {
          Out << " -> ";
          if (RetReq.isSubstitutionFailure())
            Out << "<<error-type>>";
          else if (RetReq.isTypeConstraint())
            RetReq.getTypeConstraint()->print(Out, Policy);
        }
      }
    } else {
      auto *NestedReq = cast<concepts::NestedRequirement>(Req);
      Out << "requires ";
      if (NestedReq->hasInvalidConstraint())
        Out << "<<error-expression>>";
      else
        PrintExpr(NestedReq->getConstraintExpr());
    }
    Out << "; ";
  }
  Out << "}";
}

// C++ Coroutines

void DeclPrinter::VisitCoroutineBodyStmt(CoroutineBodyStmt *S) {
  Visit(S->getBody());
}

void DeclPrinter::VisitCoreturnStmt(CoreturnStmt *S) {
  Out << "co_return";
  if (S->getOperand()) {
    Out << " ";
    Visit(S->getOperand());
  }
  Out << ";";
}

void DeclPrinter::VisitCoawaitExpr(CoawaitExpr *S) {
  Out << "co_await ";
  PrintExpr(S->getOperand());
}

void DeclPrinter::VisitDependentCoawaitExpr(DependentCoawaitExpr *S) {
  Out << "co_await ";
  PrintExpr(S->getOperand());
}

void DeclPrinter::VisitCoyieldExpr(CoyieldExpr *S) {
  Out << "co_yield ";
  PrintExpr(S->getOperand());
}

// Obj-C

void DeclPrinter::VisitObjCStringLiteral(ObjCStringLiteral *Node) {
  Out << "@";
  VisitStringLiteral(Node->getString());
}

void DeclPrinter::VisitObjCBoxedExpr(ObjCBoxedExpr *E) {
  Out << "@";
  Visit(E->getSubExpr());
}

void DeclPrinter::VisitObjCArrayLiteral(ObjCArrayLiteral *E) {
  Out << "@[ ";
  ObjCArrayLiteral::child_range Ch = E->children();
  for (auto I = Ch.begin(), E = Ch.end(); I != E; ++I) {
    if (I != Ch.begin())
      Out << ", ";
    Visit(*I);
  }
  Out << " ]";
}

void DeclPrinter::VisitObjCDictionaryLiteral(ObjCDictionaryLiteral *E) {
  Out << "@{ ";
  for (unsigned I = 0, N = E->getNumElements(); I != N; ++I) {
    if (I > 0)
      Out << ", ";

    ObjCDictionaryElement Element = E->getKeyValueElement(I);
    Visit(Element.Key);
    Out << " : ";
    Visit(Element.Value);
    if (Element.isPackExpansion())
      Out << "...";
  }
  Out << " }";
}

void DeclPrinter::VisitObjCEncodeExpr(ObjCEncodeExpr *Node) {
  Out << "@encode(";
  Node->getEncodedType().print(Out, Policy);
  Out << ')';
}

void DeclPrinter::VisitObjCSelectorExpr(ObjCSelectorExpr *Node) {
  Out << "@selector(";
  Node->getSelector().print(Out);
  Out << ')';
}

void DeclPrinter::VisitObjCProtocolExpr(ObjCProtocolExpr *Node) {
  Out << "@protocol(" << *Node->getProtocol() << ')';
}

void DeclPrinter::VisitObjCMessageExpr(ObjCMessageExpr *Mess) {
  Out << "[";
  switch (Mess->getReceiverKind()) {
  case ObjCMessageExpr::Instance:
    PrintExpr(Mess->getInstanceReceiver());
    break;

  case ObjCMessageExpr::Class:
    Mess->getClassReceiver().print(Out, Policy);
    break;

  case ObjCMessageExpr::SuperInstance:
  case ObjCMessageExpr::SuperClass:
    Out << "Super";
    break;
  }

  Out << ' ';
  Selector selector = Mess->getSelector();
  if (selector.isUnarySelector()) {
    Out << selector.getNameForSlot(0);
  } else {
    for (unsigned i = 0, e = Mess->getNumArgs(); i != e; ++i) {
      if (i < selector.getNumArgs()) {
        if (i > 0) Out << ' ';
        if (selector.getIdentifierInfoForSlot(i))
          Out << selector.getIdentifierInfoForSlot(i)->getName() << ':';
        else
           Out << ":";
      }
      else Out << ", "; // Handle variadic methods.

      PrintExpr(Mess->getArg(i));
    }
  }
  Out << "]";
}

void DeclPrinter::VisitObjCBoolLiteralExpr(ObjCBoolLiteralExpr *Node) {
  Out << (Node->getValue() ? "__objc_yes" : "__objc_no");
}

void
DeclPrinter::VisitObjCIndirectCopyRestoreExpr(ObjCIndirectCopyRestoreExpr *E) {
  PrintExpr(E->getSubExpr());
}

void
DeclPrinter::VisitObjCBridgedCastExpr(ObjCBridgedCastExpr *E) {
  Out << '(' << E->getBridgeKindName();
  E->getType().print(Out, Policy);
  Out << ')';
  PrintExpr(E->getSubExpr());
}

void DeclPrinter::VisitBlockExpr(BlockExpr *Node) {
  BlockDecl *BD = Node->getBlockDecl();
  Out << "^";

  const FunctionType *AFT = Node->getFunctionType();

  if (isa<FunctionNoProtoType>(AFT)) {
    Out << "()";
  } else if (!BD->param_empty() || cast<FunctionProtoType>(AFT)->isVariadic()) {
    Out << '(';
    for (BlockDecl::param_iterator AI = BD->param_begin(),
         E = BD->param_end(); AI != E; ++AI) {
      if (AI != BD->param_begin()) Out << ", ";
      std::string ParamStr = (*AI)->getNameAsString();
      (*AI)->getType().print(Out, Policy, ParamStr);
    }

    const auto *FT = cast<FunctionProtoType>(AFT);
    if (FT->isVariadic()) {
      if (!BD->param_empty()) Out << ", ";
      Out << "...";
    }
    Out << ')';
  }
  Out << "{ }";
}

void DeclPrinter::VisitOpaqueValueExpr(OpaqueValueExpr *Node) {
  PrintExpr(Node->getSourceExpr());
}

void DeclPrinter::VisitTypoExpr(TypoExpr *Node) {
  // TODO: Print something reasonable for a TypoExpr, if necessary.
  llvm_unreachable("Cannot print TypoExpr nodes");
}

void DeclPrinter::VisitRecoveryExpr(RecoveryExpr *Node) {
  Out << "<recovery-expr>(";
  const char *Sep = "";
  for (Expr *E : Node->subExpressions()) {
    Out << Sep;
    PrintExpr(E);
    Sep = ", ";
  }
  Out << ')';
}

void DeclPrinter::VisitAsTypeExpr(AsTypeExpr *Node) {
  Out << "__builtin_astype(";
  PrintExpr(Node->getSrcExpr());
  Out << ", ";
  Node->getType().print(Out, Policy);
  Out << ")";
}

} // end namespace

/**
 * 存储计算结构信息类实现
 */
dacppTranslator::Calc::Calc() {
  calcLoc = nullptr;
  calcBodyStmt = nullptr;
  calcCompoundStmt = nullptr;
}

void dacppTranslator::Calc::setName(std::string name) {
    this->name = name;
}
    
std::string dacppTranslator::Calc::getName() {
    return name;
}

void dacppTranslator::Calc::setParam(Param* param) {
    params.push_back(param);
}

dacppTranslator::Param* dacppTranslator::Calc::getParam(int idx) {
    return params[idx];
}

int dacppTranslator::Calc::getNumParams() {
    return params.size();
}

void dacppTranslator::Calc::setBody(Stmt* body) {
  calcBodyStmt = body;
  calcCompoundStmt = dyn_cast_or_null<CompoundStmt>(body);

  std::string Msg;
  llvm::raw_string_ostream SS(Msg);
  const ASTContext *Context = nullptr;
  DeclPrinter V(this, SS, nullptr, PrintingPolicy(LangOptions()), *Context, 0,
                "\n", false);
  V.Visit(body);
  this->body.push_back(Msg);
}

std::string dacfor_func(const std::string& code, const std::string& name) {
   std::string result = code;

    // 替换二维访问：name[i][j]
    std::regex pattern2D(R"(\b)" + name + R"(\s*\[\s*(\d+)\s*\]\s*\[\s*(\d+)\s*\])");
    std::string replacement2D = "virtual_to_physical(" + name + ", " + name + "_index+$1*info_" + name + "_acc[1]+$2, " + name + "_params)";
    result = std::regex_replace(result, pattern2D, replacement2D);

    // 替换一维访问：name[i]
    std::regex pattern1D(R"(\b)" + name + R"(\s*\[\s*(\d+)\s*\])");
    std::string replacement1D = "virtual_to_physical(" + name + ", " + name + "_index+$1, " + name + "_params)";
    result = std::regex_replace(result, pattern1D, replacement1D);
    // printf("result: %s\n", result.c_str());
    return result;
}

std::string func(std::string code, const std::string& name) {
    std::string result = code;
    // 替换二维访问 name[i][j] -> name[i*info_name_acc[1]+j]
    std::regex pattern2D(R"(\b)" + name + R"(\s*\[\s*(\d+)\s*\]\s*\[\s*(\d+)\s*\])");

    std::string replacement2D = name + "[$1*info_" + name + "_acc[1]+$2]";

    result = std::regex_replace(result, pattern2D, replacement2D);

    return result;
}

std::string dacppTranslator::Calc::getBody(int idx) {
  std::string code = body[idx];
  for (int i = 0; i < getNumParams(); i++) {

    std::string name = getParam(i)->getName();

    //std::regex pattern(name + R"((?![\w]|\[))");
    //code = std::regex_replace(code, pattern, name + "[0]");
    code = func(code, name);
    // printf("""code: %s\n", code.c_str());
  }
  return code;
}
std::string dacppTranslator::Calc::dacfor_getBody(int idx) {
    std::string code = body[idx];
    for (int i = 0; i < getNumParams(); i++) {
      std::string name = getParam(i)->getName();
      // printf("name: %s\n", name.c_str());
      // if (getParam(i)->getType().find("Tensor") != -1 ||
      //     getParam(i)->getType().find("Vector") != -1 || 
      //     getParam(i)->getType().find("Matrix") != -1) {
      //   continue;
      // }
      
      code = dacfor_func(code, name);
    }
    return code;
}

int dacppTranslator::Calc::getNumBody() {
    return body.size();
}

Stmt* dacppTranslator::Calc::getCalcBodyStmt() {
  return calcBodyStmt;
}

const Stmt* dacppTranslator::Calc::getCalcBodyStmt() const {
  return calcBodyStmt;
}

CompoundStmt* dacppTranslator::Calc::getCalcCompoundStmt() {
  return calcCompoundStmt;
}

const CompoundStmt* dacppTranslator::Calc::getCalcCompoundStmt() const {
  return calcCompoundStmt;
}

FunctionDecl* dacppTranslator::Calc::getCalcFunctionDecl() {
  return calcLoc;
}

const FunctionDecl* dacppTranslator::Calc::getCalcFunctionDecl() const {
  return calcLoc;
}

void dacppTranslator::Calc::setExpr(const BinaryOperator* dacExpr, std::vector<std::vector<int>> shapes) {
    Shell* shell = new Shell();
    Calc* calc = new Calc();
    Expression* expr = new Expression();
    expr->setDacExpr(dacExpr);
    expr->setShell(shell);
    expr->setCalc(calc);
    shell->setFather(expr);
    calc->setFather(expr);
    shell->parseShell(dacExpr, shapes);
    calc->parseCalc(dacExpr);
    exprs.push_back(expr);
}

dacppTranslator::Expression* dacppTranslator::Calc::getExpr(int idx) {
    return exprs[idx];
}

int dacppTranslator::Calc::getNumExprs() {
    return exprs.size();
}

void dacppTranslator::Calc::setFather(Expression* expr) {
    this->father = expr;
}

dacppTranslator::Expression* dacppTranslator::Calc::getFather() {
    return father;
}

void dacppTranslator::Calc::setCalcLoc(FunctionDecl* calcLoc) {
    this->calcLoc = calcLoc;
  if (calcLoc) {
    calcBodyStmt = calcLoc->getBody();
    calcCompoundStmt = dyn_cast_or_null<CompoundStmt>(calcBodyStmt);
  } else {
    calcBodyStmt = nullptr;
    calcCompoundStmt = nullptr;
  }
}
FunctionDecl* dacppTranslator::Calc::getCalcLoc() {
    return calcLoc;
}

// 解析Calc节点，将解析到的信息存储到Calc类中
void dacppTranslator::Calc::parseCalc(const BinaryOperator* dacExpr) {
    Shell* shell = getFather()->getShell();

    // 获取 DAC 数据关联表达式右值
    Expr* dacExprRHS = dacppTranslator::Expression::shellLHS_p (dacExpr) ? dacExpr->getRHS() : dacExpr->getLHS();
    FunctionDecl* calcFunc = dyn_cast<FunctionDecl>(dyn_cast<DeclRefExpr>(dacExprRHS)->getDecl());
    
    // 设置 AST 中 Calc 节点的位置
    setCalcLoc(calcFunc);

    // 设置 Calc 函数名称
    setName(calcFunc->getNameAsString());

    // 设置 Calc 参数列表
    for(unsigned int paramsCount = 0; paramsCount < calcFunc->getNumParams(); paramsCount++) {
        Param* param = new Param();

        // 获取参数读写属性
        param->setRw(inputOrOutput(calcFunc->getParamDecl(paramsCount)->getType().getAsString()));

        // 设置参数类型
        param->setType(calcFunc->getParamDecl(paramsCount)->getType());
        
        // 设置参数名称
        param->setName(calcFunc->getParamDecl(paramsCount)->getNameAsString());

        // 设置参数形状
        ShellParam* shellParam = shell->getShellParam(paramsCount);

        for (int i = 0; i < shellParam->getNumSplit(); i++) {
            Split* sp = shellParam->getSplit(i);
            if(sp->type.compare("RegularSplit") == 0) {
                RegularSplit* rsp = static_cast<RegularSplit*>(sp);
                param->setShape(rsp->getSplitSize());
            } else if(sp->type.compare("IndexSplit") == 0) {
                IndexSplit* isp = static_cast<IndexSplit*>(sp);
            } else {
                // param->setShape(shellParam->getShape(i));
            }
        }
        if (param->getDim() == 0) {
          param->setShape(1);
        }
        setParam(param);
    }
    setBody(calcFunc->getBody());

    std::string functionBody = this->body[0];
    functionBody = functionBody.substr(functionBody.find("{") + 1, functionBody.find("}") - 1);
    int end = functionBody.find("@Expression;");
    while (end != -1) {
      this->blocks.push_back(functionBody.substr(0, end));
      this->blocks.push_back("@Expression;");
      functionBody.erase(functionBody.begin(), functionBody.begin() + end + 12);
      end = functionBody.find("@Expression");
    }
    this->blocks.push_back(functionBody);
}
