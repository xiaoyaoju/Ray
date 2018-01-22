#include <cstring>
#include <cstdio>
#if defined (_WIN32)
#include <windows.h>
#endif

#include "Command.h"
#include "GoBoard.h"
#include "Gtp.h"
#include "Train.h"
#include "PatternHash.h"
#include "Rating.h"
#include "Semeai.h"
#include "UctRating.h"
#include "UctSearch.h"
#include "ZobristHash.h"
#include "OpeningBook.h"


int
main( int argc, char **argv )
{
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
  // コマンドライン引数の解析  
  AnalyzeCommand(argc, argv);

  // 各種初期化
  InitializeConst();
  InitializeRating();
  InitializeUctRating();
  InitializeUctSearch();
  InitializeSearchSetting();
  InitializeHash();
  InitializeUctHash();
  SetNeighbor();
  LoadOpeningBook(pure_board_size);

#if 0
  // GTP
  GTP_main();
#else
  Train();
#endif

  return 0;
}
