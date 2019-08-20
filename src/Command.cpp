#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "Command.h"
#include "DynamicKomi.h"
#include "GoBoard.h"
#include "Gtp.h"
#include "Train.h"
#include "Message.h"
#include "UctSearch.h"
#include "ZobristHash.h"

using namespace std;


////////////
//  定数  //
////////////

//  コマンド
const string command[COMMAND_MAX] = {
  "--playout",
  "--time",
  "--size",
  "--const-time",
  "--thread",
  "--komi",
  "--handicap",
  "--reuse-subtree",
  "--pondering",
  "--tree-size",
  "--no-debug",
  "--superko",
  "--sim-move",
  "--no-early-pass",
  "--no-nn",
  "--no-gpu",
  "--no-expand",
  "--device-id",
  "--verbose",
  "--kifu-dir",
  "--train",
  "--weights",
};

//  コマンドの説明
const string errmessage[COMMAND_MAX] = {
  "Set playouts",
  "Set all thinking time",
  "Set board size",
  "Set mode const time, and set thinking time per move",
  "Set threads",
  "Set komi",
  "Set the number of handicap stones (for testing)",
  "Reuse subtree",
  "Set pondering mode",
  "Set tree size (tree size must be 2 ^ n)",
  "Prohibit any debug message",
  "Prohibit superko move",
  "Play simulation move",
  "No early pass",
  "Don't use NN",
  "Don't use GPU",
  "No MCTS",
  "Set GPU to use",
  "Verbose log mode",
  "Set directory to store kifu files",
  "Train parameters",
  "Set location of weights file",
};

static RUN_MODE run_mode = RUN_MODE::GTP;


//////////////////////
//  コマンドの処理  //
//////////////////////
void
AnalyzeCommand( int argc, char **argv )
{
  int n, size;
  
  for (int i = 1; i < argc; i++){
    n = COMMAND_MAX + 1;
    for (int j = 0; j < COMMAND_MAX; j++){
      if (!strcmp(argv[i], command[j].c_str())){
	n = j;
      }
    }

    switch (n) {
      case COMMAND_PLAYOUT:
	// プレイアウト数固定の探索の設定
	SetPlayout(atoi(argv[++i]));
	SetMode(CONST_PLAYOUT_MODE);
	break;
      case COMMAND_TIME:
	// 持ち時間の設定
	SetTime(atof(argv[++i]));
	SetMode(TIME_SETTING_MODE);
	break;
      case COMMAND_SIZE:
	// 碁盤の大きさの設定
	size = atoi(argv[++i]);
	if (pure_board_size != size &&
	    size > 0 && size <= PURE_BOARD_SIZE) {
	  SetBoardSize(size);
	  SetParameter();
	}
	break;
      case COMMAND_CONST_TIME:
	// 1手あたりの思考時間を固定した探索の設定
	SetConstTime(atof(argv[++i]));
	SetMode(CONST_TIME_MODE);
	break;
      case COMMAND_THREAD:
	// 探索スレッド数の設定
	SetThread(atoi(argv[++i]));
	break;
      case COMMAND_KOMI:
	// コミの設定
	SetKomi(atof(argv[++i]));
	break;
      case COMMAND_HANDICAP:
	// 置き石の個数の設定
	SetConstHandicapNum(atoi(argv[++i]));
	SetHandicapNum(0);
	break;
      case COMMAND_REUSE_SUBTREE:
	// 探索結果の再利用の設定
        SetReuseSubtree(true);
        break;
      case COMMAND_PONDERING :
	// 予測読みの設定
	SetReuseSubtree(true);
	SetPonderingMode(true);
	break;
      case COMMAND_TREE_SIZE:
	// UCTのノードの個数の設定
	SetHashSize((unsigned int)atoi(argv[++i]));
	break;
      case COMMAND_SUPERKO:
        // 超劫の判定の設定
        SetSuperKo(true);
        break;
      case COMMAND_NO_DEBUG:
        // デバッグメッセージを出力しない設定
        SetDebugMessageMode(false);
        break;
      case COMMAND_SIM_MOVE:
	SetSimMove(true);
        break;
      case COMMAND_NO_EARLY_PASS:
	SetEarlyPass(false);
	break;
      case COMMAND_NO_NN:
	SetUseNN(false);
	break;
      case COMMAND_NO_GPU:
	SetDeviceId(-1);
	break;
      case COMMAND_NO_EXPAND:
        SetNoExpand(true);
        break;
      case COMMAND_DEVICE_ID:
        SetDeviceId(atoi(argv[++i]));
        break;
      case COMMAND_VERBOSE:
        SetVerbose(true);
        break;
      case COMMAND_KIFU_DIR:
        SetKifuDirectory(argv[++i]);
        break;
      case COMMAND_TRAIN:
        run_mode = RUN_MODE::TRAIN;
        break;
      case COMMAND_WEIGHTS_FILE:
        nn_model_file = argv[++i];
        break;
      default:
	for (int j = 0; j < COMMAND_MAX; j++){
	  fprintf(stderr, "%-22s : %s\n", command[j].c_str(), errmessage[j].c_str());
	}
	exit(1);
    }
  }

}


RUN_MODE
GetRunMode()
{
  return run_mode;
}
