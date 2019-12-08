#include <cstring>
#include <cstdio>
#if defined (_WIN32)
#include <windows.h>
#endif

#define BOOST_PYTHON_STATIC_LIB
#define BOOST_NUMPY_STATIC_LIB
#include <boost/python/numpy.hpp>

#include "GoBoard.h"
//#include "Gtp.h"
#include "OpeningBook.h"
#include "PatternHash.h"
#include "Rating.h"
#include "Semeai.h"
#include "Train.h"
#include "UctRating.h"
//#include "UctSearch.h"
#include "ZobristHash.h"


namespace py = boost::python;
namespace numpy = boost::python::numpy;


void collect_features(numpy::ndarray src, numpy::ndarray dst) {
}

BOOST_PYTHON_MODULE(callcfrompy) {
  Py_Initialize();
  numpy::initialize();

  char program_path[1024];
  int last;

  // 実行ファイルのあるディレクトリのパスを抽出
#if defined (_WIN32)
  HMODULE hModule = GetModuleHandle(NULL);
  GetModuleFileNameA(hModule, program_path, 1024);
#else
  strcpy(program_path, argv[0]);
#endif
  last = (int)strlen(program_path);
  while (last--){
#if defined (_WIN32)
    if (program_path[last] == '\\' || program_path[last] == '/') {
      program_path[last] = '\0';
      break;
    }
#else
    if (program_path[last] == '/') {
      program_path[last] = '\0';
      break;
    }
#endif
  }

  // 各種パスの設定
#if defined (_WIN32)
  sprintf_s(uct_params_path, 1024, "%s\\uct_params", program_path);
  sprintf_s(po_params_path, 1024, "%s\\sim_params", program_path);
#else
  snprintf(uct_params_path, 1024, "%s/uct_params", program_path);
  snprintf(po_params_path, 1024, "%s/sim_params", program_path);
#endif

  // 各種初期化
  InitializeConst();
  InitializeRating();
  InitializeUctRating();
  InitializeHash();
  InitializeUctHash();
  SetNeighbor();

  py::def("collect_features", collect_features);
}
