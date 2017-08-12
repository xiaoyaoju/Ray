#ifndef _UCTSEARCH_H_
#define _UCTSEARCH_H_

#include <atomic>
#include <random>
#include <memory>

#include "GoBoard.h"
#include "ZobristHash.h"

class LGR;
class LGRContext;

////////////
//  定数  //
////////////

const int MAX_NODES = 1000000;          // UCTのノードの配列のサイズ
const double ALL_THINKING_TIME = 90.0;  // 持ち時間(デフォルト)
const int CONST_PLAYOUT = 10000;        // 1手あたりのプレイアウト回数(デフォルト)
const double CONST_TIME = 10.0;         // 1手あたりの思考時間(デフォルト)
const int PLAYOUT_SPEED = 1000;         // 初期盤面におけるプレイアウト速度

// 思考時間の割り振り
const int TIME_RATE_9 = 20;
const int TIME_C_13 = 30;
const int TIME_MAXPLY_13 = 30;
const int TIME_C_19 = 60;
const int TIME_MAXPLY_19 = 80;

// CriticalityとOwnerを計算する間隔
const int CRITICALITY_INTERVAL = 100;

// 先頭打着緊急度
const double FPU = 5.0;

// Progressive Widening
const double PROGRESSIVE_WIDENING = 1.8;

// ノード展開の閾値
const int EXPAND_THRESHOLD_9  = 20;
const int EXPAND_THRESHOLD_13 = 25;
//const int EXPAND_THRESHOLD_19 = 40;
const int EXPAND_THRESHOLD_19 = 15;
extern int custom_expand_threshold;


// 候補手の最大数(盤上全体 + パス)
const int UCT_CHILD_MAX = PURE_BOARD_MAX + 1;

// 未展開のノードのインデックス
const int NOT_EXPANDED = -1;

// パスのインデックス
const int PASS_INDEX = 0;

// UCB Bonusに関する定数
const double BONUS_EQUIVALENCE = 1000;
const double BONUS_WEIGHT = 0.35;

// パスする勝率の閾値
const double PASS_THRESHOLD = 0.90;
// 投了する勝率の閾値
extern double resign_threshold;

// Virtual Loss (Best Parameter)
const int VIRTUAL_LOSS = 1;

extern double c_puct;
extern double value_scale;
extern double policy_temperature;
extern double policy_temperature_inc;

extern double policy_top_rate_max;
extern double seach_threshold_policy_rate;
extern double root_policy_rate_min;

extern double pass_po_limit;
extern int policy_batch_size;
extern int value_batch_size;


enum SEARCH_MODE {
  CONST_PLAYOUT_MODE,             // 1手のプレイアウト回数を固定したモード
  CONST_TIME_MODE,                // 1手の思考時間を固定したモード
  TIME_SETTING_MODE,              // 持ち時間ありのモード(秒読みなし)
  TIME_SETTING_WITH_BYOYOMI_MODE, // 持ち時間ありのモード(秒読みあり)
};


//////////////
//  構造体  //
//////////////
struct thread_arg_t {
  game_info_t *game; // 探索対象の局面
  int thread_id;   // スレッド識別番号
  int color;       // 探索する手番
};

struct statistic_t {
  std::atomic<int> colors[3];  // その箇所を領地にした回数
};

struct child_node_t {
  int pos;  // 着手する座標
  std::atomic<int> move_count;  // 探索回数
  std::atomic<int> win;         // 勝った回数
  std::atomic<bool> eval_value;
  int index;   // インデックス
  double rate; // 着手のレート
  double nnrate0; // ニューラルネットワークでのレート
  double nnrate; // ニューラルネットワークでのレート
  std::atomic<double> value;
  bool flag;   // Progressive Wideningのフラグ
  bool open;   // 常に探索候補に入れるかどうかのフラグ
  bool ladder; // シチョウのフラグ
};

//  9x9  : 1828bytes
// 13x13 : 3764bytes
// 19x19 : 7988bytes
struct uct_node_t {
  int previous_move1;                 // 1手前の着手
  int previous_move2;                 // 2手前の着手
  std::atomic<int> move_count;
  std::atomic<int> win;
  int width;                          // 探索幅
  int child_num;                      // 子ノードの数
  child_node_t child[UCT_CHILD_MAX];  // 子ノードの情報
  statistic_t statistic[BOARD_MAX];   // 統計情報 
  bool seki[BOARD_MAX];
  bool evaled;
  //std::atomic<double> value;
  std::atomic<int> value_move_count;
  std::atomic<double> value_win;
};

struct po_info_t {
  int num;   // 次の手の探索回数
  int halt;  // 探索を打ち切る回数
  std::atomic<int> count;       // 現在の探索回数
};

struct rate_order_t {
  int index;    // ノードのインデックス
  double rate;  // その手のレート
};

struct value_eval_req {
  child_node_t *uct_child;
  int color;
  int trans;
  std::vector<int> path;
  std::vector<float> data_basic;
  std::vector<float> data_features;
  std::vector<float> data_history;
};

struct policy_eval_req {
  int index;
  int depth;
  int color;
  int trans;
  std::vector<float> data_basic;
  std::vector<float> data_features;
  std::vector<float> data_history;
};


//////////////////////
//  グローバル変数  //
//////////////////////

// 残り時間
extern double remaining_time[S_MAX];
// UCTのノード
extern uct_node_t *uct_node;

// 現在のルートのインデックス
extern int current_root;

// 各座標のCriticality
extern double criticality[BOARD_MAX]; 


////////////
//  関数  //
////////////

// 予測読みを止める
void StopPondering( void );

// 予測読みのモードの設定
void SetPonderingMode( bool flag );

// 探索のモードの指定
void SetMode( enum SEARCH_MODE mode );

// 1手あたりのプレイアウト回数の指定
void SetPlayout( int po );

// 1手あたりの思考時間の指定
void SetConstTime( double time );

// 使用するスレッド数の指定
void SetThread( int new_thread );

// 持ち時間の指定
void SetTime( double time );

// 相手がパスしたらパスする
void SetEarlyPass( bool pass );

// ノード展開の有無指定
void SetNoExpand(bool flag);

//
void ToggleLiveBestSequence();

// パラメータの設定
void SetParameter( void );

// time_settingsコマンドによる設定
void SetTimeSettings( int main_time, int byoyomi, int stones );

// UCT探索の初期設定
void InitializeUctSearch( void ); 

// 探索設定の初期化
void InitializeSearchSetting( void );

// UCT探索による着手生成
int UctSearchGenmove( game_info_t *game, int color );

// 予測よみ
void UctSearchPondering( game_info_t *game, int color );

//
void UctSearchStat(game_info_t *game, int color, int num);

// UCT探索による着手生成
int UctAnalyze( game_info_t *game, int color );

// 領地になる確率をdestにコピーする
void OwnerCopy( int *dest );

// Criticaltyをdestに
void CopyCriticality( double *dest );

void CopyStatistic( statistic_t *dest );

// UCT探索による着手生成(Clean Upモード)
int UctSearchGenmoveCleanUp( game_info_t *game, int color );

// 探索の再利用の設定
void SetReuseSubtree( bool flag );


void SetUseNN(bool flag);

void SetDeviceId(int id);

// Policy networkの手を打つ
int PolicyNetworkGenmove(game_info_t *game, int color);

void EvalPolicy(const std::vector<std::shared_ptr<policy_eval_req>>& requests,
  std::vector<float>& data_basic, std::vector<float>& data_features, std::vector<float>& data_history,
  std::vector<float>& data_color, std::vector<float>& data_komi);

void EvalValue(const std::vector<std::shared_ptr<value_eval_req>>& requests,
  std::vector<float>& data_basic, std::vector<float>& data_features, std::vector<float>& data_history,
  std::vector<float>& data_color, std::vector<float>& data_komi, std::vector<float>& data_safety);

#endif
