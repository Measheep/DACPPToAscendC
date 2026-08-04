#include <string>
#include <iostream>

#include "clang/AST/AST.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "llvm/ADT/StringExtras.h"

#include "ASTParse.h"
#include "DacppStructure.h"
#include "Param.h"
#include "Shell.h"


typedef struct ArcNode
{
    int adjvex; /* 该弧所指向的顶点的位置 */
    struct ArcNode *nextarc; /* 指向下一条弧的指针 */
    char *offset;
}ArcNode; /* 表结点 */

struct VNode {
    int id;
    clang::ValueDecl *D;
    dacppTranslator::Split *s;
    ArcNode *firstarc; /* 第一个表结点的地址,指向第一条依附该顶点的弧的指针 */
} ;

struct ALGraph {
    /* table of entries.  */
    VNode *vertices;

    /* number of actual entries entered in the table.  */
    int vexnum;

    /* number of entries allocated currently.  */
    int allocated;
} ;

static int LocateVex(ALGraph *G,const clang::ValueDecl *u)
{ /* 初始条件: 图G存在,u和G中顶点有相同特征 */
  /* 操作结果: 若G中存在顶点u,则返回该顶点在图中位置;否则返回-1 */
  int i;
  for(i=0;i<G->vexnum;++i)
    if(u == G->vertices[i].D)
      return i;
  return -1;
}

static void DestroyGraph(ALGraph *G)
{ /* 初始条件: 图G存在。操作结果: 销毁图G */
  int i;
  ArcNode *p,*q;
  G->vexnum = G->allocated = 0;
  for(i=0;i<(*G).vexnum;++i)
  {
    p=(*G).vertices[i].firstarc;
    while(p)
    {
      q=p->nextarc;
      free ((void *)p->offset);
      free(p);
      p=q;
    }
  }
  free(G);
}

static ALGraph *CreateGraph(void)
{ /* 采用邻接表存储结构,构造没有相关信息的图G*/
  ALGraph *G = (ALGraph*) malloc (sizeof (struct ALGraph));
  memset (G, 0, sizeof (*G));
  return G;
}

static VNode* GetVex(ALGraph *G,int v)
{ /* 初始条件: 图G存在,v是G中某个顶点的序号。操作结果: 返回v的值 */
  return &G->vertices[v];
}

static int InsertVex(ALGraph *G,clang::ValueDecl *v, dacppTranslator::Split *s)
{ /* 初始条件: 图G存在,v和图中顶点有相同特征 */
  /* 操作结果: 在图G中增添新顶点v(不增添与顶点相关的弧,留待InsertArc()去做) */
  if (G->allocated <= G->vexnum)
  { 
    G->vertices = (VNode *)realloc (G->vertices, (1 + G->allocated) * sizeof (VNode));
    G->allocated += 1;
  }
  (*G).vertices[(*G).vexnum].D = v;
  (*G).vertices[(*G).vexnum].s = s;
  (*G).vertices[(*G).vexnum].id = (*G).vexnum;
  (*G).vertices[(*G).vexnum].firstarc=NULL;
  (*G).vexnum++; /* 图G的顶点数加1 */
  return (*G).vexnum - 1;
}

static void InsertArc(ALGraph *G,int i,int j, const char *offset)
{ /* 初始条件: 图G存在,v和w是G中两个顶点 */
  /* 操作结果: 在G中增添弧<v,w>,若G是无向的,则还增添对称弧<w,v> */
  ArcNode *p;
  p=(ArcNode*)malloc(sizeof(ArcNode));
  p->offset = (char *) malloc (sizeof (char) *
         (strlen (offset) + 1));
  strcpy (p->offset, offset);
  p->adjvex=j;
  p->nextarc=(*G).vertices[i].firstarc; /* 插在表头 */
  (*G).vertices[i].firstarc=p;
}


/**
 * 存储划分结构信息类实现
 */
dacppTranslator::Shell::Shell() {
    this->G = CreateGraph ();
}

dacppTranslator::Shell::~Shell() {
    DestroyGraph (this->G);
}

void dacppTranslator::Shell::setName(std::string name) {
    this->name = name;
}

std::string dacppTranslator::Shell::getName() {
    return name;
}

void dacppTranslator::Shell::setParam(Param* param) {
    params.push_back(param);
}

dacppTranslator::Param* dacppTranslator::Shell::getParam(int idx) {
    return params[idx];
}

int dacppTranslator::Shell::getNumParams() {
    return params.size();
}

void dacppTranslator::Shell::setSplit(Split* split) {
    splits.push_back(split);
}

dacppTranslator::Split* dacppTranslator::Shell::getSplit(int idx) {
    return splits[idx];
}

int dacppTranslator::Shell::getNumSplits() {
    return splits.size();
}

void dacppTranslator::Shell::setShellParam(ShellParam* param) {
    shellParams.push_back(param);
}

dacppTranslator::ShellParam* dacppTranslator::Shell::getShellParam(int idx) {
    return shellParams[idx];
}

int dacppTranslator::Shell::getNumShellParams() {
    return shellParams.size();
}

void dacppTranslator::Shell::setFather(Expression* expr) {
    this->father = expr;
}

dacppTranslator::Expression* dacppTranslator::Shell::getFather() {
    return father;
}

void dacppTranslator::Shell::setShellLoc(FunctionDecl* shellLoc) {
    this->shellLoc = shellLoc;
}

FunctionDecl* dacppTranslator::Shell::getShellLoc() {
    return shellLoc;
}

struct Visitor : RecursiveASTVisitor<Visitor> {
    dacppTranslator::Shell *sh;
    const BinaryOperator* dacExpr;
  
    Visitor(dacppTranslator::Shell *sh, const BinaryOperator* dacExpr)
        : sh(sh), dacExpr(dacExpr) {}

    Expr *ignoreImplicitSemaNodes(Expr *Node) {
        Expr *E = Node;
        while (true) {
          if (auto *MTE = dyn_cast<MaterializeTemporaryExpr>(E))
            E = MTE->getSubExpr();
          if (auto *BTE = dyn_cast<CXXBindTemporaryExpr>(E))
            E = BTE->getSubExpr();
          if (auto *ICE = dyn_cast<ImplicitCastExpr>(E))
            E = ICE->getSubExpr();
          else
            break;
        }
    
        return E;
    }

    bool VisitVarDecl(VarDecl *D);
    bool VisitCallExpr(CallExpr *Call);
};

bool Visitor::VisitCallExpr(CallExpr *Call) {
  std::string TempString;
  llvm::raw_string_ostream SS(TempString);
  unsigned i, e;
  int v1 = -1,v2 = -1;
  std::string offset1, offset2;

  Call->getCallee()->printPretty(SS, nullptr, PrintingPolicy(LangOptions()));
  /* 如果是一个binding函数，就进行解析。  */
  if (!strcmp(SS.str().c_str(), "binding")) {
    for (i = 0, e = Call->getNumArgs(); i != e; ++i) {
      if (isa<CXXDefaultArgExpr>(Call->getArg(i))) {
        // Don't print any defaulted arguments
        break;
      }
      if (Expr *tempExpr = ignoreImplicitSemaNodes(Call->getArg(i))) {
        /* 不带偏移量。  */
        if (CXXConstructExpr *CCE = dyn_cast<CXXConstructExpr>(tempExpr)) {
          if (CCE->getNumArgs() == 1) {
            tempExpr = ignoreImplicitSemaNodes(CCE->getArg(0));
            if (const auto *DeclRef = dyn_cast<DeclRefExpr>(tempExpr))
              /* 应该总是能找到。  */
              (i % 2 ? v2 : v1) = LocateVex (sh->G, DeclRef->getDecl());
          }
        }
        
        /* 带偏移量。  */
        else if (auto *OpCallExpr = dyn_cast<CXXOperatorCallExpr>(tempExpr)) {
          /* ‘+’/‘-’有两个参数。 */
          if (OpCallExpr->getNumArgs() == 2) {
            llvm::raw_string_ostream Buf(i % 2 ? offset2 : offset1);
            tempExpr = ignoreImplicitSemaNodes(OpCallExpr->getArg(0));
            if (const auto *DeclRef = dyn_cast<DeclRefExpr>(tempExpr))
              /* 应该总是能找到。  */
              (i % 2 ? v2 : v1) = LocateVex (sh->G, DeclRef->getDecl());
            /* 加数/减数 */
            Buf << ' ' << getOperatorSpelling(OpCallExpr->getOperator()) << ' ';
            OpCallExpr->getArg(1)->printPretty(Buf, nullptr, PrintingPolicy(LangOptions()));
          }
        }
      }
    }
    if (v1 != v2) {
      /* 如果被加数/被减数带偏移量，则移项 */
      if (offset1.c_str()[0])
        offset2 += " - (" + offset1 + ")";
      /* 插入边 */
      InsertArc(sh->G, v1, v2, offset2.c_str());
    }
  }
  return true;
}

bool Visitor::VisitVarDecl (VarDecl *D)
{
  VarDecl *curVarDecl = D;
  do
  /*
      只解析两种节点
      1.算子(目前只有降维算子以及规则分区算子)
      2.dacpp::list
  */
  {
    if (curVarDecl->getType().getAsString().compare("dacpp::list") != 0 &&
        curVarDecl->getType().getAsString().compare("dacpp::index") != 0 &&
        curVarDecl->getType().getAsString().compare("dacpp::split") != 0) {
      break;
    }

    // 解析降维算子
    if (curVarDecl->getType().getAsString().compare("dacpp::index") == 0) {
      dacppTranslator::IndexSplit *sp = new dacppTranslator::IndexSplit(nullptr);
      sp->setId(curVarDecl->getNameAsString());
      sp->type = "IndexSplit";
      sh->setSplit(sp);
      if (-1 == LocateVex(sh->G, curVarDecl)) {
        sp->v = GetVex (sh->G, InsertVex (sh->G, curVarDecl, sp));
      }
      break;
    }

    // 解析规则分区算子
    if (curVarDecl->getType().getAsString().compare("dacpp::split") ==
        0) {
      dacppTranslator::RegularSplit *sp = new dacppTranslator::RegularSplit(nullptr);
      sp->setId(curVarDecl->getNameAsString());
      CXXConstructExpr *CCE = nullptr;
      if (isa<CXXConstructExpr>(curVarDecl->getInit())) {
        CCE = dyn_cast<CXXConstructExpr>(curVarDecl->getInit());
      } else {
        CCE = dacppTranslator::getNode<CXXConstructExpr>(curVarDecl->getInit());
      }
      int count = 0;
      for (CXXConstructExpr::arg_iterator I = CCE->arg_begin(),
                                          E = CCE->arg_end();
           I != E; ++I) {
        if (count == 0) {
          /* TODO: 计算常量表达式的值。  */
          sp->setSplitSize(std::stoi(
              toString((dyn_cast<IntegerLiteral>(*I))->getValue(), 10, true)));
        } else if (count == 1) {
          /* TODO: 计算常量表达式的值。  */
          sp->setSplitStride(std::stoi(
              toString((dyn_cast<IntegerLiteral>(*I))->getValue(), 10, true)));
        }
        count++;
      }
      sp->type = "RegularSplit";
      sh->setSplit(sp);
      if (-1 == LocateVex (sh->G, curVarDecl)) {
        sp->v = GetVex (sh->G, InsertVex (sh->G, curVarDecl, sp));
      }
      break;
    }

    /*
        解析dacpp::list
        对数据的所有划分信息均在dacpp::list的初始化列表中
    */
    InitListExpr *ILE =
        dacppTranslator::getNode<InitListExpr>(curVarDecl->getInit());
    for (unsigned int i = 0; i < ILE->getNumInits(); i++) {
      dacppTranslator::ShellParam *shellParam = new dacppTranslator::ShellParam();
      Expr *curExpr = ILE->getInit(i);
      std::vector<Expr *> astExprs;

      std::string name = "";
      dacppTranslator::getSplitExpr(curExpr, name, astExprs);
      shellParam->setName(name);
      for (unsigned int paramsCount = 0;
           paramsCount < sh->getShellLoc()->getNumParams(); paramsCount++) {
        if (shellParam->getName() !=
            sh->getShellLoc()->getParamDecl(paramsCount)->getNameAsString()) {
          continue;
        }
        shellParam->setRw(
            dacppTranslator::inputOrOutput(sh->getShellLoc()
                                               ->getParamDecl(paramsCount)
                                               ->getType()
                                               .getAsString()));
        shellParam->setType(sh->getParam(paramsCount)->newType);
        shellParam->setName(sh->getParam(paramsCount)->getName());
        for (int shapeIdx = 0; shapeIdx < sh->getParam(paramsCount)->getDim(); shapeIdx++) {
            shellParam->setShape(sh->getParam(paramsCount)->getShape(shapeIdx));
        }
      }
      for (unsigned int i = 0; i < astExprs.size(); i++) {
        if (dacppTranslator::getNode<DeclRefExpr>(astExprs[i])) {
          VarDecl *vd = dyn_cast<VarDecl>(
              dacppTranslator::getNode<DeclRefExpr>(astExprs[i])->getDecl());

          if (vd->getType().getAsString().compare("dacpp::split") == 0) {
            dacppTranslator::RegularSplit *sp =
                new dacppTranslator::RegularSplit((dacppTranslator::RegularSplit *) GetVex(sh->G, LocateVex (sh->G, vd))->s);
            sp->type = "RegularSplit";
            sp->setId(vd->getNameAsString());
            sp->setDimIdx(i);
            CXXConstructExpr *CCE = nullptr;
            if (isa<CXXConstructExpr>(vd->getInit())) {
              CCE = dyn_cast<CXXConstructExpr>(vd->getInit());
            } else {
                dacppTranslator::getNode<CXXConstructExpr>(vd->getInit());
            }
            int count = 0;
            for (CXXConstructExpr::arg_iterator I = CCE->arg_begin(),
                                                E = CCE->arg_end();
                 I != E; ++I) {
              if (count == 0) {
                /* TODO: 计算常量表达式的值。  */
                sp->setSplitSize(std::stoi(toString(
                    (dyn_cast<IntegerLiteral>(*I))->getValue(), 10, true)));
              } else if (count == 1) {
                /* TODO: 计算常量表达式的值。  */
                sp->setSplitStride(std::stoi(toString(
                    (dyn_cast<IntegerLiteral>(*I))->getValue(), 10, true)));
              }
              count++;
            }
            /* TODO: 处理除不尽的情况。  */
            // sp->setSplitNumber((shellParam->getShape(i) - sp->getSplitSize()) / sp->getSplitStride() + 1);
            for (int m = 0; m < sh->getNumSplits(); m++) {
              if (sh->getSplit(m)->getId().compare(sp->getId()) == 0 &&
                  sh->getSplit(m)->type.compare("RegularSplit") == 0) {
                dacppTranslator::RegularSplit *isp =
                    static_cast<dacppTranslator::RegularSplit *>(
                        sh->getSplit(m));
                isp->setSplitNumber(sp->getSplitNumber());
              }
            }
            shellParam->setSplit(sp);
          } else if (vd->getType().getAsString().compare("dacpp::index") == 0) {
            dacppTranslator::IndexSplit *sp = new dacppTranslator::IndexSplit((dacppTranslator::IndexSplit *) GetVex(sh->G, LocateVex (sh->G, vd))->s);
            sp->type = "IndexSplit";
            sp->setId(vd->getNameAsString());
            sp->setDimIdx(i);
            // sp->setSplitNumber(shellParam->getShape(i));
            for (int m = 0; m < sh->getNumSplits(); m++) {
              if (sh->getSplit(m)->getId().compare(sp->getId()) == 0 &&
                  sh->getSplit(m)->type.compare("IndexSplit") == 0) {
                dacppTranslator::IndexSplit *isp =
                    static_cast<dacppTranslator::IndexSplit *>(sh->getSplit(m));
                isp->setSplitNumber(sp->getSplitNumber());
              }
            }
            shellParam->setSplit(sp);
          }
        } else {
          dacppTranslator::Split *sp = new dacppTranslator::Split(nullptr);
          sp->type = "Split";
          sp->setId("void");
          sp->setDimIdx(i);
          shellParam->setSplit(sp);
        }
      }
      sh->setShellParam(shellParam);
    }
  } while (0);

  return true;
}

void dacppTranslator::Shell::GetBindInfo(
    std::vector<BINDINFO> *pbindInfo)
{ /*按广度优先非递归遍历图G。使用辅助队列Q和访问标志数组visited。*/
  bool *visited;
  std::queue<BINDINFO *> Q;
  int v;
  BINDINFO bindinfo ;
  std::string parent;
  int icls = 0;
  ArcNode *p;
  int *refs;

  pbindInfo->clear();
  visited = (bool *)malloc(sizeof(bool) * G->vexnum);
  memset(visited, 0, sizeof(bool) * G->vexnum);
  refs = (int *)malloc(sizeof(int) * G->vexnum);
  memset(refs, 0, sizeof(int) * G->vexnum);

  for(v=0;v<G->vexnum;v++) /* 如果是连通图,只v=0就遍历全图 */
    for (p = G->vertices[v].firstarc; p; p = p->nextarc)
      refs[p->adjvex]++;

  for(v=0;v<G->vexnum;v++) /* 如果是连通图,只v=0就遍历全图 */
  {
    if(!visited[v] && !refs[v]) /* v尚未访问 */
    {
      icls++;
      visited[v]=true;
      bindinfo.icls = icls;
      bindinfo.v = GetVex (G, v);
      bindinfo.offset = "";
      pbindInfo->push_back(bindinfo);
      Q.push (&bindinfo);
      while(!Q.empty()) /* 队列不空 */
      {
        bindinfo = *Q.front();
        parent = bindinfo.offset;
        Q.pop();
        for (p = G->vertices[bindinfo.v->id].firstarc; p; p = p->nextarc)
        {
          if(!visited[p->adjvex]) /* 尚未访问的邻接顶点 */
          {
            visited[p->adjvex]=true;
            bindinfo.icls = icls;
            bindinfo.v = GetVex (G, p->adjvex);
            bindinfo.offset = parent + p->offset;
            pbindInfo->push_back(bindinfo);
            Q.push (&bindinfo); /* 入队 */
          }
        }
      }
    }
  }
  free(refs);
  free(visited);
}

dacppTranslator::Split *dacppTranslator::Shell::search_symbol(VNode *v) {
  dacppTranslator::Split *s = v->s;
  while (s->parent) s = s->parent;
  return s;
}


// 解析Shell节点，将解析到的信息存储到Shell类中
void dacppTranslator::Shell::parseShell(const BinaryOperator* dacExpr, std::vector<std::vector<int>> shapes) {
  std::string Msg;
  llvm::raw_string_ostream SS(Msg);
    Visitor V (this, dacExpr);

  /*
      数据关联计算表达式节点为一个BinaryOperator节点
      左值为CallExpr节点，表示shell函数调用
      右值为DeclRefExpr，表示calc函数引用
  */
  Expr *dacExprLHS = dacppTranslator::Expression::shellLHS_p (dacExpr) ? dacExpr->getLHS() : dacExpr->getRHS();
  CallExpr *shellCall = getNode<CallExpr>(dacExprLHS);
  FunctionDecl *shellFunc = shellCall->getDirectCallee();

  // 设置 AST 中 Shell 节点的位置
  setShellLoc(shellFunc);

  // 设置 Shell 函数名称
  setName(shellFunc->getNameAsString());

  // 设置 Shell 参数列表
  for (unsigned int paramsCount = 0; paramsCount < shellFunc->getNumParams();
       paramsCount++) {
    Param *param = new Param();

    // 获取参数读写属性
    param->setRw(inputOrOutput(
        shellFunc->getParamDecl(paramsCount)->getType().getAsString()));

    // 设置参数类型
    param->setType(shellFunc->getParamDecl(paramsCount)->getType());

    // 设置参数名称
    param->setName(shellFunc->getParamDecl(paramsCount)->getNameAsString());

    // 设置参数形状
    if (paramsCount < shapes.size()) {
      for (unsigned int i = 0; i < shapes[paramsCount].size(); i++) {
        param->setShape(shapes[paramsCount][i]);
      }
    }

    setParam(param);
    }

    // 获取shell函数体
    Stmt* shellFuncBody = shellFunc->getBody();
    V.TraverseStmt (shellFuncBody);

    std::vector<BINDINFO> bindInfo;
    GetBindInfo (&bindInfo);
}
