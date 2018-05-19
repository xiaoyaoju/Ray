#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <mutex>
#include <numeric>
#include <thread>
#include <random>
#include <queue>

#include "DynamicKomi.h"
#include "GoBoard.h"
#include "Ladder.h"
#include "Message.h"
#include "MoveCache.h"
#include "PatternHash.h"
#include "Point.h"
#include "Rating.h"
#include "Seki.h"
#include "Simulation.h"
#include "UctRating.h"
#include "UctSearch.h"
#include "Utility.h"

#if defined (_WIN32)
#define NOMINMAX
#include <Windows.h>
#else
#include <unistd.h>
#include <pthread.h>
#include <semaphore.h>
#include <sys/time.h>
#endif

#include "CNTKLibrary.h"

using namespace std;

#define LOCK_NODE(var) mutex_nodes[(var)].lock()
#define UNLOCK_NODE(var) mutex_nodes[(var)].unlock()
#define LOCK_EXPAND mutex_expand.lock();
#define UNLOCK_EXPAND mutex_expand.unlock();

typedef std::pair<std::wstring, std::vector<float>*> MapEntry;
typedef std::map<std::wstring, std::vector<float>*> Layer;

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

void ReadWeights();
void EvalNode();
//void EvalUctNode(std::vector<int>& indices, std::vector<int>& color, std::vector<int>& trans, std::vector<float>& data, std::vector<int>& path);

////////////////
//  大域変数  //
////////////////

// 持ち時間
double remaining_time[S_MAX];

// UCTのノード
uct_node_t *uct_node;

// プレイアウト情報
static po_info_t po_info;

// Progressive Widening の閾値
static int pw[PURE_BOARD_MAX + 1];

// ノード展開の閾値
static int expand_threshold = EXPAND_THRESHOLD_19;

static double value_evaluation_threshold = 0;

// ノードを展開しない
static bool no_expand = false;

// 試行時間を延長するかどうかのフラグ
static bool extend_time = false;

int current_root; // 現在のルートのインデックス
mutex mutex_nodes[MAX_NODES];
mutex mutex_expand;       // ノード展開を排他処理するためのmutex

mutex mutex_queue;

// 探索の設定
enum SEARCH_MODE mode = CONST_TIME_MODE;
// 使用するスレッド数
int threads = 1;
// 1手あたりの試行時間
double const_thinking_time = CONST_TIME;
// 1手当たりのプレイアウト数
int playout = CONST_PLAYOUT;
// デフォルトの持ち時間
double default_remaining_time = ALL_THINKING_TIME;

// 各スレッドに渡す引数
vector<thread_arg_t> t_arg;

// プレイアウトの統計情報
statistic_t statistic[BOARD_MAX];
// 盤上の各点のCriticality
double criticality[BOARD_MAX];
// 盤上の各点のOwner(0-100%)
static double owner[BOARD_MAX];

// 現在のオーナーのインデックス
int owner_index[BOARD_MAX];
// 現在のクリティカリティのインデックス
int criticality_index[BOARD_MAX];

// 候補手のフラグ
bool candidates[BOARD_MAX];

// 投了する勝率の閾値
double resign_threshold = 0.20;

bool pondering_mode = false;

bool ponder = false;

bool pondering_stop = false;

bool pondered = false;

double time_limit;

std::vector<std::unique_ptr<std::thread>> handle;    // スレッドのハンドル

static volatile bool running;

// UCB Bonusの等価パラメータ
double bonus_equivalence = BONUS_EQUIVALENCE;
// UCB Bonusの重み
double bonus_weight = BONUS_WEIGHT;

// 乱数生成器
std::vector<std::unique_ptr<std::mt19937_64>> mt;

// Last-Good-Reply
LGR lgr;
std::vector<LGRContext> lgr_ctx;

// Criticalityの上限値
int criticality_max = CRITICALITY_MAX;

//
bool reuse_subtree = false;

// 自分の手番の色
int my_color;

//
static bool live_best_sequence = false;

double policy_temperature = 0.49;
double policy_temperature_inc = 0.056;
double c_puct = 2;
double value_scale = 0.8;
int custom_expand_threshold = -1;

double policy_top_rate_max = 0.90;
double seach_threshold_policy_rate = 0.005;

double pass_po_limit = 0.5;
int policy_batch_size = 16;
int value_batch_size = 64;

ray_clock::time_point begin_time;

static bool early_pass = true;

static bool use_nn = true;
static int device_id = -2;
static std::queue<std::shared_ptr<policy_eval_req>> eval_policy_queue;
static std::queue<std::shared_ptr<value_eval_req>> eval_value_queue;
static int eval_count_policy, eval_count_value;
static double owner_nn[BOARD_MAX];

static CNTK::FunctionPtr nn_policy;
static CNTK::FunctionPtr nn_value;

//template<double>
double atomic_fetch_add(std::atomic<double> *obj, double arg) {
  double expected = obj->load();
  while (!atomic_compare_exchange_weak(obj, &expected, expected + arg))
    ;
  return expected;
}

static void
ClearEvalQueue()
{
  mutex_queue.lock();
  queue<shared_ptr<value_eval_req>> empty_value;
  eval_value_queue.swap(empty_value);
  queue<shared_ptr<policy_eval_req>> empty_policy;
  eval_policy_queue.swap(empty_policy);
  mutex_queue.unlock();
}

////////////
//  関数  //
////////////

// Virtual Lossを加算
static int AddVirtualLoss( child_node_t *child, int current );

// 次のプレイアウト回数の設定
static void CalculateNextPlayouts( game_info_t *game, int color, double best_wp, double finish_time );

// Criticaliityの計算
static void CalculateCriticality( int color );

// Criticality
static void CalculateCriticalityIndex( uct_node_t *node, statistic_t *node_statistic, int color, int *index );

// Ownershipの計算
static void CalculateOwner( int color, int count );

// Ownership
static void CalculateOwnerIndex( uct_node_t *node, statistic_t *node_statistc, int color, int *index );

// 現局面の子ノードのインデックスの導出
static void CorrectDescendentNodes( vector<int> &indexes, int index );

// ノードの展開
static int ExpandNode( game_info_t *game, int color, int current, const std::vector<int>& path );

// ルートの展開
static int ExpandRoot( game_info_t *game, int color );

// 思考時間を延長する処理
static bool ExtendTime( void );

// 候補手の初期化
static void InitializeCandidate( child_node_t *uct_child, int pos, bool ladder );

// 探索打ち切りの確認
static bool InterruptionCheck( void );

// UCT探索
static void ParallelUctSearch( thread_arg_t *arg );

// UCT探索(予測読み)
static void ParallelUctSearchPondering( thread_arg_t *arg );

// ノードのレーティング
static void RatingNode( game_info_t *game, int color, int index, int depth );

static int RateComp( const void *a, const void *b );

// UCB値が最大の子ノードを返す
static int SelectMaxUcbChild(const game_info_t *game, int current, int color );

// 各座標の統計処理
static void Statistic( game_info_t *game, int winner );

// UCT探索(1回の呼び出しにつき, 1回の探索)
static int UctSearch(game_info_t *game, int color, mt19937_64 *mt, LGR& lgrf, LGRContext& lgrctx, int current, int *winner, std::vector<int>& path);

// 各ノードの統計情報の更新
static void UpdateNodeStatistic( game_info_t *game, int winner, statistic_t *node_statistic );

// 結果の更新
static void UpdateResult( child_node_t *child, int result, int current );

// 乱数の初期化
static void InitRand();

/////////////////////
//  予測読みの設定  //
/////////////////////
void
SetPonderingMode( bool flag )
{
  pondering_mode = flag;
}


////////////////////////
//  探索モードの指定  //
////////////////////////
void
SetMode( enum SEARCH_MODE new_mode )
{
  mode = new_mode;
}


///////////////////////////////////////
//  1手あたりのプレイアウト数の指定  //
///////////////////////////////////////
void
SetPlayout( int po )
{
  playout = po;
}


/////////////////////////////////
//  1手にかける試行時間の設定  //
/////////////////////////////////
void
SetConstTime( double time )
{
  const_thinking_time = time;
}


////////////////////////////////
//  使用するスレッド数の指定  //
////////////////////////////////
void
SetThread( int new_thread )
{
  threads = new_thread;

  lgr_ctx.resize(threads);

  InitRand();
}


//////////////////////
//  持ち時間の設定  //
//////////////////////
void
SetTime( double time )
{
  default_remaining_time = time;
}


//////////////////////////
//  ノード再利用の設定  //
//////////////////////////
void
SetReuseSubtree( bool flag )
{
  reuse_subtree = flag;
}

//////////////////
//  パスの設定  //
//////////////////
void
SetEarlyPass(bool pass)
{
  early_pass = pass;
}

//////////////////////////////
// Toggle Live Best Sequece //
//////////////////////////////
void
ToggleLiveBestSequence()
{
  live_best_sequence = !live_best_sequence;
}

////////////////////////////////////////////
//  盤の大きさに合わせたパラメータの設定  //
////////////////////////////////////////////
void
SetParameter( void )
{
  if (custom_expand_threshold > 0) {
    expand_threshold = custom_expand_threshold;
  }  else if (pure_board_size < 11) {
    expand_threshold = EXPAND_THRESHOLD_9;
  } else if (pure_board_size < 16) {
    expand_threshold = EXPAND_THRESHOLD_13;
  } else {
    expand_threshold = EXPAND_THRESHOLD_19;
  }
}

////////////////////
//  NN利用の設定  //
////////////////////
void
SetUseNN(bool flag)
{
  use_nn = flag;
}

void
SetDeviceId(const int id)
{
  device_id = id;
}

void
SetNoExpand(bool flag)
{
  no_expand = flag;
}

//////////////////////////////////////
//  time_settingsコマンドによる設定  //
//////////////////////////////////////
void
SetTimeSettings( int main_time, int byoyomi, int stone )
{
  if (mode == CONST_PLAYOUT_MODE ||
      mode == CONST_TIME_MODE) {
    return ;
  }

  if (main_time == 0) {
    const_thinking_time = (double)byoyomi * 0.85;
    mode = CONST_TIME_MODE;
    cerr << "Const Thinking Time Mode" << endl;
  } else {
    if (byoyomi == 0) {
      default_remaining_time = main_time;
      mode = TIME_SETTING_MODE;
      cerr << "Time Setting Mode" << endl;
    } else {
      default_remaining_time = main_time;
      const_thinking_time = ((double)byoyomi) / stone;
      mode = TIME_SETTING_WITH_BYOYOMI_MODE;
      cerr << "Time Setting Mode (byoyomi)" << endl;
    }
  }
}

////////////////
// 乱数の初期化 //
////////////////
static void
InitRand()
{
  mt.clear();
  random_device rd;
  for (int i = 0; i < threads; i++) {
    mt.push_back(make_unique<mt19937_64>(rd()));
  }
}


/////////////////////////
//  UCT探索の初期設定  //
/////////////////////////
void
InitializeUctSearch( void )
{
  int i;

  // Progressive Wideningの初期化
  pw[0] = 0;
  for (i = 1; i <= PURE_BOARD_MAX; i++) {
    pw[i] = pw[i - 1] + (int)(40 * pow(PROGRESSIVE_WIDENING, i - 1));
    if (pw[i] > 10000000) break;
  }
  for (i = i + 1; i <= PURE_BOARD_MAX; i++) {
    pw[i] = INT_MAX;
  }

  // UCTのノードのメモリを確保
  uct_node = new uct_node_t[uct_hash_size];

  if (uct_node == NULL) {
    cerr << "Cannot allocate memory !!" << endl;
    cerr << "You must reduce tree size !!" << endl;
    exit(1);
  }

  if (use_nn && !nn_policy)
    ReadWeights();
}


////////////////////////
//  探索設定の初期化  //
////////////////////////
void
InitializeSearchSetting( void )
{
  // Ownerの初期化
  for (int i = 0; i < board_max; i++){
    owner[i] = 50;
    owner_index[i] = 5;
    candidates[i] = true;
  }

  // 乱数の初期化
  InitRand();

  // Initialize Last-Good-Reply
  lgr.reset();
  lgr_ctx.resize(threads);

  // 持ち時間の初期化
  for (int i = 0; i < 3; i++) {
    remaining_time[i] = default_remaining_time;
  }

  // 制限時間を設定
  // プレイアウト回数の初期化
  if (mode == CONST_PLAYOUT_MODE) {
    time_limit = 100000.0;
    po_info.num = playout;
    extend_time = false;
  } else if (mode == CONST_TIME_MODE) {
    time_limit = const_thinking_time;
    po_info.num = 100000000;
    extend_time = false;
  } else if (mode == TIME_SETTING_MODE ||
	     mode == TIME_SETTING_WITH_BYOYOMI_MODE) {
    if (pure_board_size < 11) {
      time_limit = remaining_time[0] / TIME_RATE_9;
      po_info.num = (int)(PLAYOUT_SPEED * time_limit);
      extend_time = true;
    } else if (pure_board_size < 13) {
      time_limit = remaining_time[0] / (TIME_MAXPLY_13 + TIME_C_13);
      po_info.num = (int)(PLAYOUT_SPEED * time_limit);
      extend_time = true;
    } else {
      time_limit = remaining_time[0] / (TIME_MAXPLY_19 + TIME_C_19);
      po_info.num = (int)(PLAYOUT_SPEED * time_limit);
      extend_time = true;
    }
  }

  pondered = false;
  pondering_stop = true;
}


void
StopPondering( void )
{
  if (!pondering_mode) {
    return ;
  }

  if (ponder) {
    pondering_stop = true;
    for (auto& t : handle) {
      t->join();
    }
    handle.clear();

    ponder = false;
    pondered = true;
    PrintPonderingCount(po_info.count);
  }
}


/////////////////////////////////////
//  UCTアルゴリズムによる着手生成  //
/////////////////////////////////////
int
UctSearchGenmove( game_info_t *game, int color )
{
  int pos, select_index, max_count, pre_simulated;
  double finish_time, pass_wp, best_wp;
  child_node_t *uct_child;

  // 探索情報をクリア
  if (!pondered) {
    memset(statistic, 0, sizeof(statistic_t) * board_max);
    fill_n(criticality_index, board_max, 0);
    for (int i = 0; i < board_max; i++) {
      criticality[i] = 0.0;
    }
  }
  po_info.count = 0;

  for (int i = 0; i < pure_board_max; i++) {
    pos = onboard_pos[i];
    owner[pos] = 50;
    owner_index[pos] = 5;
    candidates[pos] = true;

    owner_nn[pos] = 50;
  }

  if (!reuse_subtree) {
    ClearUctHash();
  }

  ClearEvalQueue();

  eval_count_policy = 0;
  eval_count_value = 0;

  // 探索開始時刻の記録
  begin_time = ray_clock::now();

  // UCTの初期化
  current_root = ExpandRoot(game, color);

  // 前回から持ち込んだ探索回数を記録
  pre_simulated = uct_node[current_root].move_count;

  // 子ノードが1つ(パスのみ)ならPASSを返す
  if (uct_node[current_root].child_num <= 1) {
    return PASS;
  }

  // 探索回数の閾値を設定
  po_info.halt = po_info.num;

  // 自分の手番を設定
  my_color = color;

  // Dynamic Komiの算出(置碁のときのみ)
  DynamicKomi(game, &uct_node[current_root], color);

  // 探索時間とプレイアウト回数の予定値を出力
  PrintPlayoutLimits(time_limit, po_info.halt);

  t_arg.resize(threads);
  running = true;
  for (int i = 0; i < threads; i++) {
    t_arg[i].thread_id = i;
    t_arg[i].game = game;
    t_arg[i].color = color;
    handle.push_back(make_unique<thread>(ParallelUctSearch, &t_arg[i]));
  }

  if (use_nn) {
    handle.push_back(make_unique<thread>(EvalNode));
  }

  for (auto &t : handle) {
    t->join();
  }
  handle.clear();

  // 着手が41手以降で,
  // 時間延長を行う設定になっていて,
  // 探索時間延長をすべきときは
  // 探索回数を1.5倍に増やす
  if (game->moves > pure_board_size * 3 - 17 &&
      extend_time &&
      ExtendTime()) {
    po_info.halt = (int)(1.5 * po_info.halt);
    time_limit *= 1.5;
    running = true;
    for (int i = 0; i < threads; i++) {
      handle.push_back(make_unique<thread>(ParallelUctSearch, &t_arg[i]));
    }

    if (use_nn) {
      handle.push_back(make_unique<thread>(EvalNode));
    }

    for (auto &t : handle) {
      t->join();
    }
    handle.clear();
  }

  uct_child = uct_node[current_root].child;

  select_index = PASS_INDEX;
  max_count = early_pass ? (int)uct_child[PASS_INDEX].move_count : 0;

  // 探索回数最大の手を見つける
  for (int i = 1; i < uct_node[current_root].child_num; i++){
    if (uct_child[i].move_count > max_count) {
      select_index = i;
      max_count = uct_child[i].move_count;
    }
  }

  // 探索にかかった時間を求める
  finish_time = GetSpendTime(begin_time);

  // パスの勝率の算出
  if (uct_child[PASS_INDEX].move_count != 0) {
    pass_wp = (double)uct_child[PASS_INDEX].win / uct_child[PASS_INDEX].move_count;
  } else {
    pass_wp = 0;
  }

  // 選択した着手の勝率の算出(Dynamic Komi)
  best_wp = (double)uct_child[select_index].win / uct_child[select_index].move_count;
  double best_wpv = (double)uct_node[current_root].value_win / uct_node[current_root].value_move_count;

  // コミを含めない盤面のスコアを求める
  game_info_t game_copy;
  CopyGame(&game_copy, game);
  double score = (double)CalculateScore(&game_copy);
  // コミを考慮した勝敗
  score -= komi[my_color];

  // 各地点の領地になる確率の出力
  PrintOwner(&uct_node[current_root], color, owner);

  // 取れている石を数える
  int count = 0;
  for (int i = 0; i < pure_board_max; i++) {
    int pos = onboard_pos[i];

    if (game->board[pos] == FLIP_COLOR(color) && owner[pos] > 70) {
      count++;
    }
  }

  // パスをするときは
  // 1. 直前の着手がパスで, パスした時の勝率がPASS_THRESHOLD以上
  //    early_pass か有効か死石をすべて打ち上げ済み
  // 2. 着手数がMAX_MOVES以上
  // 投了するときは
  //    Dynamic Komiでの勝率がresign_threshold以下
  // それ以外は選ばれた着手を返す
  if (pass_wp >= PASS_THRESHOLD &&
      (early_pass || count == 0) &&
      (game->record[game->moves - 1].pos == PASS)){
    pos = PASS;
  } else if (game->moves >= MAX_MOVES) {
    pos = PASS;
  } else if (game->moves > 3 &&
             early_pass &&
	     game->record[game->moves - 1].pos == PASS &&
	     game->record[game->moves - 3].pos == PASS) {
    pos = PASS;
  } else if (!early_pass && count == 0 && best_wp < pass_wp && max_count < uct_child[PASS_INDEX].move_count) {
    pos = PASS;
  } else if (best_wp <= resign_threshold && (!use_nn || best_wpv < resign_threshold)) {
    pos = RESIGN;
  } else {
    pos = uct_child[select_index].pos;
  }

  // 最善応手列を出力
  PrintBestSequence(game, uct_node, current_root, color);
  // 探索の情報を出力(探索回数, 勝敗, 思考時間, 勝率, 探索速度)
  PrintPlayoutInformation(&uct_node[current_root], &po_info, finish_time, pre_simulated);
  // 次の探索でのプレイアウト回数の算出
  CalculateNextPlayouts(game, color, best_wp, finish_time);

  ClearEvalQueue();

  if (use_nn && GetDebugMessageMode()) {
    cerr << "Eval NN Policy     :  " << setw(7) << (eval_count_policy + eval_policy_queue.size()) << endl;
    cerr << "Eval NN Value      :  " << setw(7) << (eval_count_value + eval_value_queue.size()) << endl;
    cerr << "Eval NN            :  " << setw(7) << eval_count_policy << "/" << eval_count_value << "/" << value_evaluation_threshold << endl;
    cerr << "Count Captured     :  " << setw(7) << count << endl;
    cerr << "Score              :  " << setw(7) << score << endl;
    PrintMoveStat(cerr, game, uct_node, current_root);
    //PrintOwnerNN(S_BLACK, owner_nn);
  }

  return pos;
}


///////////////
//  予測読み  //
///////////////
void
UctSearchPondering(game_info_t *game, int color)
{
  if (!pondering_mode) {
    return;
  }

  // 探索情報をクリア
  memset(statistic, 0, sizeof(statistic_t) * board_max);
  fill_n(criticality_index, board_max, 0);
  for (int i = 0; i < board_max; i++) {
    criticality[i] = 0.0;
  }

  po_info.count = 0;

  for (int i = 0; i < pure_board_max; i++) {
    const int pos = onboard_pos[i];
    owner[pos] = 50;
    owner_index[pos] = 5;
    candidates[pos] = true;
  }

  // UCTの初期化
  current_root = ExpandRoot(game, color);

  pondered = false;

  // 子ノードが1つ(パスのみ)ならPASSを返す
  if (uct_node[current_root].child_num <= 1) {
    ponder = false;
    pondering_stop = true;
    return;
  }

  ponder = true;
  pondering_stop = false;

  // Dynamic Komiの算出(置碁のときのみ)
  DynamicKomi(game, &uct_node[current_root], color);

  t_arg.resize(threads);
  running = true;
  for (int i = 0; i < threads; i++) {
    t_arg[i].thread_id = i;
    t_arg[i].game = game;
    t_arg[i].color = color;
    handle.push_back(make_unique<thread>(ParallelUctSearchPondering, &t_arg[i]));
  }

  if (use_nn) {
    handle.push_back(make_unique<thread>(EvalNode));
  }

  return ;
}

/////////////////////////////////////
// 統計
/////////////////////////////////////
void
UctSearchStat(game_info_t *game, int color, int num)
{
  int i, pos;
  double finish_time;
  int select_index;
  int max_count;
  double pass_wp;
  double best_wp;
  child_node_t *uct_child;
  int pre_simulated;


  // 探索情報をクリア
  memset(statistic, 0, sizeof(statistic_t) * board_max);
  memset(criticality_index, 0, sizeof(int) * board_max);
  memset(criticality, 0, sizeof(double) * board_max);
  po_info.count = 0;

  for (i = 0; i < pure_board_max; i++) {
    pos = onboard_pos[i];
    owner[pos] = 50;
    owner_index[pos] = 5;
    candidates[pos] = true;
  }

  if (reuse_subtree) {
    DeleteOldHash(game);
  } else {
    ClearUctHash();
  }

  bool org_use_nn = use_nn;
  use_nn = false;

  // 探索開始時刻の記録
  begin_time = ray_clock::now();

  // UCTの初期化
  current_root = ExpandRoot(game, color);

  // 前回から持ち込んだ探索回数を記録
  pre_simulated = uct_node[current_root].move_count;

  // 子ノードが1つ(パスのみ)ならPASSを返す
  if (uct_node[current_root].child_num <= 1) {
    return;
  }

  // 探索回数の閾値を設定
  po_info.halt = num;

  // 自分の手番を設定
  my_color = color;

  // Dynamic Komiの算出(置碁のときのみ)
  DynamicKomi(game, &uct_node[current_root], color);

  t_arg.resize(threads);
  running = true;
  for (i = 0; i < threads; i++) {
    t_arg[i].thread_id = i;
    t_arg[i].game = game;
    t_arg[i].color = color;
    handle.push_back(make_unique<thread>(ParallelUctSearch, &t_arg[i]));
  }

  for (auto &t : handle) {
    t->join();
  }
  handle.clear();

  use_nn = org_use_nn;

  uct_child = uct_node[current_root].child;

  select_index = PASS_INDEX;
  max_count = uct_child[PASS_INDEX].move_count;

  // 探索回数最大の手を見つける
  for (i = 1; i < uct_node[current_root].child_num; i++){
    if (uct_child[i].move_count > max_count) {
      select_index = i;
      max_count = uct_child[i].move_count;
    }
  }

  // 探索にかかった時間を求める
  finish_time = GetSpendTime(begin_time);

  // パスの勝率の算出
  if (uct_child[PASS_INDEX].move_count != 0) {
    pass_wp = (double)uct_child[PASS_INDEX].win / uct_child[PASS_INDEX].move_count;
  } else {
    pass_wp = 0;
  }

  // 選択した着手の勝率の算出(Dynamic Komi)
  best_wp = (double)uct_child[select_index].win / uct_child[select_index].move_count;

  cerr << (color == S_BLACK ? "BLACK" : "WHITE") << endl;
  // 各地点の領地になる確率の出力
  PrintOwner(&uct_node[current_root], color, owner);

  // 探索の情報を出力(探索回数, 勝敗, 思考時間, 勝率, 探索速度)
  PrintPlayoutInformation(&uct_node[current_root], &po_info, finish_time, 0);
}

/////////////////////
//  候補手の初期化  //
/////////////////////
static void
InitializeCandidate( child_node_t *uct_child, int pos, bool ladder )
{
  uct_child->pos = pos;
  uct_child->move_count = 0;
  uct_child->win = 0;
  uct_child->eval_value = false;
  uct_child->index = NOT_EXPANDED;
  uct_child->rate = 0.0;
  uct_child->flag = false;
  uct_child->open = false;
  uct_child->ladder = ladder;
  uct_child->nnrate = 0;
  uct_child->value = -1;
}


/////////////////////////
//  ルートノードの展開  //
/////////////////////////
static int
ExpandRoot( game_info_t *game, int color )
{
  const int moves = game->moves;
  unsigned long long hash = game->move_hash;
  unsigned int index = FindSameHashIndex(hash, color, moves);
  int pos, child_num = 0, pm1 = PASS, pm2 = PASS;
  bool ladder[BOARD_MAX] = { false };
  child_node_t *uct_child;

  // 直前の着手の座標を取り出す
  pm1 = game->record[moves - 1].pos;
  // 2手前の着手の座標を取り出す
  if (moves > 1) pm2 = game->record[moves - 2].pos;

  // 9路盤でなければシチョウを調べる
  if (pure_board_size != 9) {
    LadderExtension(game, color, ladder);
  }

  std::vector<int> path;

  // 既に展開されていた時は, 探索結果を再利用する
  if (index != uct_hash_size) {
    vector<int> indexes;

    // 現局面の子ノード以外を削除する
    CorrectDescendentNodes(indexes, index);
    std::sort(indexes.begin(), indexes.end());
    ClearNotDescendentNodes(indexes);

    // 直前と2手前の着手を更新
    uct_node[index].previous_move1 = pm1;
    uct_node[index].previous_move2 = pm2;

    uct_child = uct_node[index].child;

    child_num = uct_node[index].child_num;

    for (int i = 0; i < child_num; i++) {
      pos = uct_child[i].pos;
      uct_child[i].rate = 0.0;
      uct_child[i].flag = false;
      uct_child[i].open = false;
      if (ladder[pos]) {
	uct_node[index].move_count -= uct_child[i].move_count;
	uct_node[index].win -= uct_child[i].win;
	uct_child[i].move_count = 0;
	uct_child[i].win = 0;
	uct_child[i].eval_value = false;
      }
      uct_child[i].ladder = ladder[pos];
    }

    path.push_back(index);

    // 展開されたノード数を1に初期化
    uct_node[index].width = 1;

    // 候補手のレーティング
    RatingNode(game, color, index, path.size());

    PrintReuseCount(uct_node[index].move_count);

    return index;
  } else {
    // 全ノードのクリア
    ClearUctHash();

    // 空のインデックスを探す
    index = SearchEmptyIndex(hash, color, moves);

    assert(index != uct_hash_size);

    // ルートノードの初期化
    uct_node[index].previous_move1 = pm1;
    uct_node[index].previous_move2 = pm2;
    uct_node[index].move_count = 0;
    uct_node[index].win = 0;
    uct_node[index].width = 0;
    uct_node[index].child_num = 0;
    uct_node[index].evaled = false;
    uct_node[index].value_move_count = 0;
    uct_node[index].value_win = 0;
    memset(uct_node[index].statistic, 0, sizeof(statistic_t) * BOARD_MAX);
    fill_n(uct_node[index].seki, BOARD_MAX, false);

    uct_child = uct_node[index].child;

    // パスノードの展開
    InitializeCandidate(&uct_child[PASS_INDEX], PASS, ladder[PASS]);
    child_num++;

    // 候補手の展開
    if (moves == 1) {
      for (int i = 0; i < first_move_candidates; i++) {
	pos = first_move_candidate[i];
	// 探索候補かつ合法手であれば探索対象にする
	if (candidates[pos] && IsLegal(game, pos, color)) {
	  InitializeCandidate(&uct_child[child_num], pos, ladder[pos]);
	  child_num++;
	}
      }
    } else {
      for (int i = 0; i < pure_board_max; i++) {
	pos = onboard_pos[i];
	// 探索候補かつ合法手であれば探索対象にする
	if (candidates[pos] && IsLegal(game, pos, color)) {
	  InitializeCandidate(&uct_child[child_num], pos, ladder[pos]);
	  child_num++;
	}
      }
    }

    path.push_back(index);

    // 子ノード個数の設定
    uct_node[index].child_num = child_num;

    // 候補手のレーティング
    RatingNode(game, color, index, path.size());

    // セキの確認
    CheckSeki(game, uct_node[index].seki);

    uct_node[index].width++;
  }

  return index;
}



///////////////////
//  ノードの展開  //
///////////////////
static int
ExpandNode( game_info_t *game, int color, int current, const std::vector<int>& path )
{
  const int moves = game->moves;
  unsigned long long hash = game->move_hash;
  unsigned int index = FindSameHashIndex(hash, color, moves);
  int child_num = 0, max_pos = PASS, sibling_num, pm1 = PASS, pm2 = PASS;
  double max_rate = 0.0;
  child_node_t *uct_child, *uct_sibling;
  bool ladder[BOARD_MAX] = { false };

  // 合流先が検知できれば, それを返す
  if (index != uct_hash_size) {
    return index;
  }

  // 空のインデックスを探す
  index = SearchEmptyIndex(hash, color, moves);

  assert(index != uct_hash_size);

  // 直前の着手の座標を取り出す
  pm1 = game->record[moves - 1].pos;
  // 2手前の着手の座標を取り出す
  if (moves > 1) pm2 = game->record[moves - 2].pos;

  // 9路盤でなければシチョウを調べる
  if (pure_board_size != 9) {
    LadderExtension(game, color, ladder);
  }

  // 現在のノードの初期化
  uct_node[index].previous_move1 = pm1;
  uct_node[index].previous_move2 = pm2;
  uct_node[index].move_count = 0;
  uct_node[index].win = 0;
  uct_node[index].width = 0;
  uct_node[index].child_num = 0;
  uct_node[index].evaled = false;
  uct_node[index].value_move_count = 0;
  uct_node[index].value_win = 0;
  memset(uct_node[index].statistic, 0, sizeof(statistic_t) * BOARD_MAX);
  fill_n(uct_node[index].seki, BOARD_MAX, false);
  uct_child = uct_node[index].child;

  // パスノードの展開
  InitializeCandidate(&uct_child[PASS_INDEX], PASS, ladder[PASS]);
  child_num++;

  // 候補手の展開
  for (int i = 0; i < pure_board_max; i++) {
    const int pos = onboard_pos[i];
    // 探索候補でなければ除外
    if (candidates[pos] && IsLegal(game, pos, color)) {
      InitializeCandidate(&uct_child[child_num], pos, ladder[pos]);
      child_num++;
    }
  }

  // 子ノードの個数を設定
  uct_node[index].child_num = child_num;

  // 候補手のレーティング
  RatingNode(game, color, index, path.size() + 1);

  // セキの確認
  CheckSeki(game, uct_node[index].seki);

  // 探索幅を1つ増やす
  uct_node[index].width++;

  // 兄弟ノードで一番レートの高い手を求める
  uct_sibling = uct_node[current].child;
  sibling_num = uct_node[current].child_num;
  for (int i = 0; i < sibling_num; i++) {
    if (uct_sibling[i].pos != pm1) {
      if (uct_sibling[i].rate > max_rate) {
	max_rate = uct_sibling[i].rate;
	max_pos = uct_sibling[i].pos;
      }
    }
  }

  // 兄弟ノードで一番レートの高い手を展開する
  for (int i = 0; i < child_num; i++) {
    if (uct_child[i].pos == max_pos) {
      if (!uct_child[i].flag) {
	uct_child[i].open = true;
      }
      break;
    }
  }

  return index;
}


//////////////////////////////////////
//  ノードのレーティング             //
//  (Progressive Wideningのために)  //
//////////////////////////////////////
static void
RatingNode( game_info_t *game, int color, int index, int depth )
{
  const int child_num = uct_node[index].child_num;
  const int moves = game->moves;
  int pos, max_index;
  int pat_index[3] = {0};
  double score = 0.0, max_score, dynamic_parameter;
  bool self_atari_flag;
  pattern_hash_t hash_pat;
  child_node_t *uct_child = uct_node[index].child;
  uct_features_t uct_features;

  memset(&uct_features, 0, sizeof(uct_features_t));

  // パスのレーティング
  uct_child[PASS_INDEX].rate = CalculateLFRScore(game, PASS, pat_index, &uct_features);

  // 直前の着手で発生した特徴の確認
  UctCheckFeatures(game, color, &uct_features);
  // 直前の着手で石を2つ取られたか確認
  UctCheckRemove2Stones(game, color, &uct_features);
  // 直前の着手で石を3つ取られたか確認
  UctCheckRemove3Stones(game, color, &uct_features);
  // 2手前で劫が発生していたら, 劫を解消するトリの確認
  if (game->ko_move == moves - 2) {
    UctCheckCaptureAfterKo(game, color, &uct_features);
    UctCheckKoConnection(game, &uct_features);
  }

  max_index = 0;
  max_score = uct_child[0].rate;

  if (use_nn) {
    //int color = game->record[game->moves - 1].color;

    uct_node_t *root = &uct_node[current_root];

    double rate[PURE_BOARD_MAX];
    AnalyzePoRating(game, color, rate);

    auto req = make_shared<policy_eval_req>();
    req->color = color;
    req->depth = depth;
    req->index = index;
    req->trans = rand() / (RAND_MAX / 8 + 1);
    //req.path.swap(path);
    WritePlanes(req->data_basic, req->data_features, req->data_history, nullptr,
      game, root, color, req->trans);
#if 1
    mutex_queue.lock();
    eval_policy_queue.push(req);
    mutex_queue.unlock();
    //push_back(u);
#else
    std::vector<int> indices;
    indices.push_back(index);
    EvalUctNode(indices, req.data);
#endif
  }

  for (int i = 1; i < child_num; i++) {
    pos = uct_child[i].pos;

    // 自己アタリの確認
    self_atari_flag = UctCheckSelfAtari(game, color, pos, &uct_features);
    // ウッテガエシの確認
    UctCheckSnapBack(game, color, pos, &uct_features);
    // トリの確認
    if ((uct_features.tactical_features1[pos] & capture_mask)== 0) {
      UctCheckCapture(game, color, pos, &uct_features);
    }
    // アタリの確認
    if ((uct_features.tactical_features1[pos] & atari_mask) == 0) {
      UctCheckAtari(game, color, pos, &uct_features);
    }
    // 両ケイマの確認
    UctCheckDoubleKeima(game, color, pos, &uct_features);
    // ケイマのツケコシの確認
    UctCheckKeimaTsukekoshi(game, color, pos, &uct_features);

    // 自己アタリが無意味だったらスコアを0.0にする
    // 逃げられないシチョウならスコアを-1.0にする
    if (!self_atari_flag) {
      score = 0.0;
    } else if (uct_child[i].ladder) {
      score = -1.0;
    } else {
#if 1
      // MD3, MD4, MD5のパターンのハッシュ値を求める
      PatternHash(&game->pat[pos], &hash_pat);
      // MD3のパターンのインデックスを探す
      pat_index[0] = SearchIndex(md3_index, hash_pat.list[MD_3]);
      // MD4のパターンのインデックスを探す
      pat_index[1] = SearchIndex(md4_index, hash_pat.list[MD_4]);
      // MD5のパターンのインデックスを探す
      pat_index[2] = SearchIndex(md5_index, hash_pat.list[MD_5 + MD_MAX]);

      score = CalculateLFRScore(game, pos, pat_index, &uct_features);
#else
      pos = uct_child[i].pos;
      int x = X(pos) - OB_SIZE;
      int y = Y(pos) - OB_SIZE;
      int n = x + y * pure_board_size;
      score = outputs[n] / sum;
      if (score > 0)
        uct_child[i].flag = true;
#endif
    }

    // その手のγを記録
    uct_child[i].rate = score;

    // 現在見ている箇所のOwnerとCriticalityの補正値を求める
    dynamic_parameter = uct_owner[owner_index[pos]] + uct_criticality[criticality_index[pos]];

    // 最もγが大きい着手を記録する
    if (score + dynamic_parameter > max_score) {
      max_index = i;
      max_score = score + dynamic_parameter;
    }
  }

  // 最もγが大きい着手を探索できるようにする
  uct_child[max_index].flag = true;
}




//////////////////////////
//  探索打ち止めの確認  //
//////////////////////////
static bool
InterruptionCheck( void )
{
  const int child_num = uct_node[current_root].child_num;
  const int rest = po_info.halt - po_info.count;
  int max = 0, second = 0;
  child_node_t *uct_child = uct_node[current_root].child;

  if (mode != CONST_PLAYOUT_MODE &&
      GetSpendTime(begin_time) * 2.0 < time_limit) {
      return false;
  }

  // 探索回数が最も多い手と次に多い手を求める
  for (int i = 0; i < child_num; i++) {
    if (uct_child[i].move_count > max) {
      second = max;
      max = uct_child[i].move_count;
    } else if (uct_child[i].move_count > second) {
      second = uct_child[i].move_count;
    }
  }

  // 残りの探索を全て次善手に費やしても
  // 最善手を超えられない場合は探索を打ち切る
  if (max - second > rest) {
    return true;
  } else {
    return false;
  }
}


///////////////////////////
//  思考時間延長の確認   //
///////////////////////////
static bool
ExtendTime( void )
{
  // 持ち時間が少ないときは延長しない
  if (time_limit < 2.0)
    return false;

  int max = 0, second = 0;
  int max_index = 0;
  const int child_num = uct_node[current_root].child_num;
  child_node_t *uct_child = uct_node[current_root].child;

  // 探索回数が最も多い手と次に多い手を求める
  for (int i = 0; i < child_num; i++) {
    if (uct_child[i].move_count > max) {
      second = max;
      max = uct_child[i].move_count;
      max_index = i;
    } else if (uct_child[i].move_count > second) {
      second = uct_child[i].move_count;
    }
  }

  // 最善手の探索回数がが次善手の探索回数の
  // 1.2倍未満なら探索延長
  if (max < second * 1.2) {
    if (GetDebugMessageMode()) {
      cerr << "Extend time "
        << FormatMove(uct_child[max_index].pos) << " max:" << max
        << " second:" << second << endl;
    }
    return true;
  }

  // Extend time if policy value is too low
  if (uct_node[current_root].evaled) {
    if (uct_child[max_index].nnrate < 0.02) {
      if (GetDebugMessageMode()) {
        cerr << "Extend time "
          << FormatMove(uct_child[max_index].pos) << " policy:" << uct_child[max_index].nnrate << endl;
      }
      return true;
    } else {
      if (GetDebugMessageMode()) {
        cerr << "Stop policy:" << uct_child[max_index].nnrate << endl;
      }
    }
  }

  return false;
}


static void
WaitForEvaluationQueue()
{
  static std::atomic<int> queue_full;

  value_evaluation_threshold = max(0.0, value_evaluation_threshold - 0.01);

  // Wait if dcnn queue is full
  mutex_queue.lock();
  while (eval_value_queue.size() > value_batch_size * 3 || eval_policy_queue.size() > policy_batch_size * 3) {
    if (GetSpendTime(begin_time) > time_limit || !running) break;
    std::atomic_fetch_add(&queue_full, 1);
    value_evaluation_threshold = min(0.5, value_evaluation_threshold + 0.01);
    mutex_queue.unlock();
    this_thread::sleep_for(chrono::milliseconds(10));
    if (queue_full % 1000 == 0)
      cerr << "EVAL QUEUE FULL" << endl;
    mutex_queue.lock();
  }
  mutex_queue.unlock();
}

/////////////////////////////////
//  並列処理で呼び出す関数     //
//  UCTアルゴリズムを反復する  //
/////////////////////////////////
static void
ParallelUctSearch( thread_arg_t *arg )
{
  thread_arg_t *targ = (thread_arg_t *)arg;
  game_info_t *game;
  int color = targ->color;
  bool interruption = false;
  bool enough_size = true;
  int winner = 0;
  int interval = CRITICALITY_INTERVAL;

  game = AllocateGame();

  // スレッドIDが0のスレッドだけ別の処理をする
  // 探索回数が閾値を超える, または探索が打ち切られたらループを抜ける
  if (targ->thread_id == 0) {
    do {
      // Wait if dcnn queue is full
      WaitForEvaluationQueue();
      // 探索回数を1回増やす
      atomic_fetch_add(&po_info.count, 1);
      // 盤面のコピー
      CopyGame(game, targ->game);
      // 1回プレイアウトする
      std::vector<int> path;
      UctSearch(game, color, mt[targ->thread_id].get(), lgr, lgr_ctx[targ->thread_id], current_root, &winner, path);
      // 探索を打ち切るか確認
      interruption = InterruptionCheck();
      // ハッシュに余裕があるか確認
      enough_size = CheckRemainingHashSize();
      // OwnerとCriticalityを計算する
      if (po_info.count > interval) {
	CalculateOwner(color, po_info.count);
	CalculateCriticality(color);
	interval += CRITICALITY_INTERVAL;
      }
      if (GetSpendTime(begin_time) > time_limit) break;
      if (!enough_size) cerr << "HASH TABLE FULL" << endl;
    } while (po_info.count < po_info.halt && !interruption && enough_size);
    running = false;
  } else {
    do {
      // Wait if dcnn queue is full
      WaitForEvaluationQueue();
      // 探索回数を1回増やす
      atomic_fetch_add(&po_info.count, 1);
      // 盤面のコピー
      CopyGame(game, targ->game);
      // 1回プレイアウトする
	  std::vector<int> path;
      UctSearch(game, color, mt[targ->thread_id].get(), lgr, lgr_ctx[targ->thread_id], current_root, &winner, path);
      // 探索を打ち切るか確認
      interruption = InterruptionCheck();
      // ハッシュに余裕があるか確認
      enough_size = CheckRemainingHashSize();
      if (GetSpendTime(begin_time) > time_limit) break;
      if (!enough_size) cerr << "HASH TABLE FULL" << endl;
    } while (po_info.count < po_info.halt && !interruption && enough_size);
  }

  // メモリの解放
  FreeGame(game);
  return;
}


/////////////////////////////////
//  並列処理で呼び出す関数     //
//  UCTアルゴリズムを反復する  //
/////////////////////////////////
static void
ParallelUctSearchPondering( thread_arg_t *arg )
{
  thread_arg_t *targ = (thread_arg_t *)arg;
  game_info_t *game;
  int color = targ->color;
  bool enough_size = true;
  int winner = 0;
  int interval = CRITICALITY_INTERVAL;

  game = AllocateGame();

  // スレッドIDが0のスレッドだけ別の処理をする
  // 探索回数が閾値を超える, または探索が打ち切られたらループを抜ける
  if (targ->thread_id == 0) {
    do {
      // Wait if dcnn queue is full
      WaitForEvaluationQueue();
      // 探索回数を1回増やす
      atomic_fetch_add(&po_info.count, 1);
      // 盤面のコピー
      CopyGame(game, targ->game);
      // 1回プレイアウトする
      std::vector<int> path;
      UctSearch(game, color, mt[targ->thread_id].get(), lgr, lgr_ctx[targ->thread_id], current_root, &winner, path);
      // ハッシュに余裕があるか確認
      enough_size = CheckRemainingHashSize();
      // OwnerとCriticalityを計算する
      if (po_info.count > interval) {
	CalculateOwner(color, po_info.count);
	CalculateCriticality(color);
	interval += CRITICALITY_INTERVAL;
      }
    } while (!pondering_stop && enough_size);
    running = false;
  } else {
    do {
      // Wait if dcnn queue is full
      WaitForEvaluationQueue();
      // 探索回数を1回増やす
      atomic_fetch_add(&po_info.count, 1);
      // 盤面のコピー
      CopyGame(game, targ->game);
      // 1回プレイアウトする
      std::vector<int> path;
      UctSearch(game, color, mt[targ->thread_id].get(), lgr, lgr_ctx[targ->thread_id], current_root, &winner, path);
      // ハッシュに余裕があるか確認
      enough_size = CheckRemainingHashSize();
    } while (!pondering_stop && enough_size);
  }

  // メモリの解放
  FreeGame(game);
  return;
}


//////////////////////////////////////////////
//  UCT探索を行う関数                        //
//  1回の呼び出しにつき, 1プレイアウトする    //
//////////////////////////////////////////////
static int
UctSearch(game_info_t *game, int color, mt19937_64 *mt, LGR& lgrf, LGRContext& lgrctx, int current, int *winner, std::vector<int>& path)
{
  int result = 0, next_index;
  double score;
  child_node_t *uct_child = uct_node[current].child;

  // 現在見ているノードをロック
  LOCK_NODE(current);
  // UCB値最大の手を求める
  next_index = SelectMaxUcbChild(game, current, color);
  // Store context hash
  {
    int child_num = uct_node[current].child_num;
    int index = -1;
    int max = 0;
    for (int i = 0; i < child_num; i++) {
      if (uct_child[i].move_count > max) {
        max = uct_child[i].move_count;
        index = i;
      }
    }
    if (index == -1 || uct_child[index].move_count < 10) {
      lgrctx.store(game, PASS);
    } else {
      lgrctx.store(game, uct_child[index].pos);
    }
  }
  // 選んだ手を着手
  PutStone(game, uct_child[next_index].pos, color);
  // 色を入れ替える
  color = FLIP_COLOR(color);

  bool end_of_game = game->moves > 2 &&
    game->record[game->moves - 1].pos == PASS &&
    game->record[game->moves - 2].pos == PASS;

  if (no_expand || uct_child[next_index].move_count < expand_threshold || end_of_game) {
    int start = game->moves;
    path.push_back(current);

    // Virtual Lossを加算
    int n = AddVirtualLoss(&uct_child[next_index], current);

    memcpy(game->seki, uct_node[current].seki, sizeof(bool) * BOARD_MAX);

    // 現在見ているノードのロックを解除
    UNLOCK_NODE(current);

    // Enqueue value

    bool expected = false;
    if (use_nn
      && (n >= expand_threshold * value_evaluation_threshold
        || mode == CONST_PLAYOUT_MODE)
      && atomic_compare_exchange_strong(&uct_child[next_index].eval_value, &expected, true)) {

      uct_node_t *root = &uct_node[current_root];

      double rate[PURE_BOARD_MAX];
      AnalyzePoRating(game, color, rate);
      auto req = make_shared<value_eval_req>();
      req->uct_child = uct_child + next_index;
      req->color = color;
      //req->index = index;
      req->trans = rand() / (RAND_MAX / 8 + 1);
      req->path.swap(path);
      WritePlanes(req->data_basic, req->data_features, req->data_history, nullptr,
        game, root, color, req->trans);
      mutex_queue.lock();
      eval_value_queue.push(req);
      mutex_queue.unlock();
    }

    // 終局まで対局のシミュレーション
    Simulation(game, color, mt, lgr, lgrctx);

    // コミを含めない盤面のスコアを求める
    score = (double)CalculateScore(game);

    // コミを考慮した勝敗
    if (my_color == S_BLACK) {
      if (score - dynamic_komi[my_color] >= 0) {
	result = (color == S_BLACK ? 0 : 1);
	*winner = S_BLACK;
      } else {
	result = (color == S_WHITE ? 0 : 1);
	*winner = S_WHITE;
      }
    } else {
      if (score - dynamic_komi[my_color] > 0) {
	result = (color == S_BLACK ? 0 : 1);
	*winner = S_BLACK;
      } else {
	result = (color == S_WHITE ? 0 : 1);
	*winner = S_WHITE;
      }
    }
    // 統計情報の記録
    Statistic(game, *winner);

    lgr.update(game, start, *winner, lgrctx);
  } else {
    path.push_back(current);
    // Virtual Lossを加算
    AddVirtualLoss(&uct_child[next_index], current);
    // ノードの展開の確認
    if (uct_child[next_index].index == -1) {
      // ノードの展開中はロック
      LOCK_EXPAND;
      // ノードの展開
      uct_child[next_index].index = ExpandNode(game, color, current, path);
      //cerr << "value evaluated " << result << " " << v << " " << *value_result << endl;
      // ノード展開のロックの解除
      UNLOCK_EXPAND;
    }
    // 現在見ているノードのロックを解除
    UNLOCK_NODE(current);
    // 手番を入れ替えて1手深く読む
    result = UctSearch(game, color, mt, lgrf, lgrctx, uct_child[next_index].index, winner, path);
    //
    // double v = uct_node[current].value;
    // if (*value_result < 0 && v >= 0) {
    //   *value_result = color == S_WHITE ? v : 1 - v;
    // }
  }
#if 0
  double v = uct_node[current].value.load();
  if (v >= 0) {
    cerr << "value evaluated " << result << " " << v << " " << *value_result << endl;
  }
  if (*value_result < 0 && v >= 0 && atomic_compare_exchange_strong(&uct_node[current].value, &v, -2.0)) {
    *value_result = color == S_BLACK ? 1 - v : v;
    cerr << "update value " << result << " " << v << endl;
  }
#endif

  // 探索結果の反映
  UpdateResult(&uct_child[next_index], result, current);

  // 統計情報の更新
  UpdateNodeStatistic(game, *winner, uct_node[current].statistic);

  // if (*value_result >= 0)
  // 	*value_result = 1 - *value_result;
  return 1 - result;
}


//////////////////////////
//  Virtual Lossの加算  //
//////////////////////////
static int
AddVirtualLoss(child_node_t *child, int current)
{
  atomic_fetch_add(&uct_node[current].move_count, VIRTUAL_LOSS);
  int org = atomic_fetch_add(&child->move_count, VIRTUAL_LOSS);
  return org;
}


//////////////////////
//  探索結果の更新  //
/////////////////////
static void
UpdateResult( child_node_t *child, int result, int current )
{
  atomic_fetch_add(&uct_node[current].win, result);
  atomic_fetch_add(&uct_node[current].move_count, 1 - VIRTUAL_LOSS);
  atomic_fetch_add(&child->win, result);
  atomic_fetch_add(&child->move_count, 1 - VIRTUAL_LOSS);
  // if (value >= 0) {
  //   atomic_fetch_add(&uct_node[current].value_win, value);
  //   atomic_fetch_add(&uct_node[current].value_move_count, 1);
  // }
}


//////////////////////////
//  ノードの並び替え用  //
//////////////////////////
static int
RateComp( const void *a, const void *b )
{
  rate_order_t *ro1 = (rate_order_t *)a;
  rate_order_t *ro2 = (rate_order_t *)b;
  if (ro1->rate < ro2->rate) {
    return 1;
  } else if (ro1->rate > ro2->rate) {
    return -1;
  } else {
    return 0;
  }
}

static void
UpdatePolicyRate(int current)
{
  const int move_count = uct_node[current].move_count;
  child_node_t *uct_child = uct_node[current].child;
  const int child_num = uct_node[current].child_num;
  double t = policy_temperature + log(move_count + 1) * policy_temperature_inc;

  double sum;
  double rate;
  int n = 0;

  double max_nnrate0 = numeric_limits<double>::min();
  for (int i = 1; i < child_num; i++) {
    max_nnrate0 = max(uct_child[i].nnrate0, max_nnrate0);
  }
  double offset = 10 - max_nnrate0;

  // TODO cleanup
  do {
    double max_rate = 0;
    double min_rate = numeric_limits<double>::max();

    sum = 0;

    for (int i = 1; i < child_num; i++) {
      double rate = exp((uct_child[i].nnrate0 + offset) / t);
      if (rate > max_rate)
        max_rate = rate;
      if (rate < min_rate)
        min_rate = rate;
      sum += rate;
    }
    rate = max_rate / sum;

    //if (n != 0) cerr << "#" << n << " max:" << (max_rate / sum) << " temperature:" << t << " " << tt << endl;
    if (rate > policy_top_rate_max)
      t *= 1.05;
    n++;
  } while (t > 0 && t < 10 && rate > policy_top_rate_max);

  for (int i = 1; i < child_num; i++) {
    double rate = exp((uct_child[i].nnrate0 + offset) / t) / sum;
    uct_child[i].nnrate = max(rate, 0.0);
  }
}


/////////////////////////////////////////////////////
//  UCBが最大となる子ノードのインデックスを返す関数  //
/////////////////////////////////////////////////////
static int
SelectMaxUcbChild( const game_info_t *game, int current, int color )
{
  bool evaled = uct_node[current].evaled;
  child_node_t *uct_child = uct_node[current].child;
  const int child_num = uct_node[current].child_num;
  const int sum = uct_node[current].move_count;
  double ucb_value;
  int max_index;
  double max_rate;
  double dynamic_parameter;
  rate_order_t order[PURE_BOARD_MAX + 1];
  int width;
  double child_ucb[UCT_CHILD_MAX];
  double child_lcb[UCT_CHILD_MAX];
  double ucb_bonus_weight = bonus_weight * sqrt(bonus_equivalence / (sum + bonus_equivalence));
  const bool debug = current == current_root && sum % 10000 == 0 && GetDebugMessageMode();

  if (live_best_sequence && current == current_root && sum % 1000 == 0) {
    PrintBestSequenceGFX(cerr, game, uct_node, current_root, color);
  }
  //if (evaled) {
    //cerr << "use nn" << endl;
//  } else
  {
    // 128回ごとにOwnerとCriticalityでソートし直す
    if ((sum & 0x7f) == 0 && sum != 0) {
      int o_index[UCT_CHILD_MAX], c_index[UCT_CHILD_MAX];
      CalculateCriticalityIndex(&uct_node[current], uct_node[current].statistic, color, c_index);
      CalculateOwnerIndex(&uct_node[current], uct_node[current].statistic, color, o_index);
      for (int i = 0; i < child_num; i++) {
	int pos = uct_child[i].pos;
	dynamic_parameter = uct_owner[o_index[i]] + uct_criticality[c_index[i]];
	order[i].rate = uct_child[i].rate + dynamic_parameter;
	order[i].index = i;
	uct_child[i].flag |= uct_child[i].nnrate > seach_threshold_policy_rate;
      }
      qsort(order, child_num, sizeof(rate_order_t), RateComp);

      // 子ノードの数と探索幅の最小値を取る
      width = ((uct_node[current].width > child_num) ? child_num : uct_node[current].width);

      // 探索候補の手を展開し直す
      for (int i = 0; i < width; i++) {
	uct_child[order[i].index].flag = true;
      }

      if (evaled && policy_temperature_inc > 0)
        UpdatePolicyRate(current);
    }

    // Progressive Wideningの閾値を超えたら,
    // レートが最大の手を読む候補を1手追加
    if (sum > pw[uct_node[current].width]) {
      max_index = -1;
      max_rate = 0;
      for (int i = 0; i < child_num; i++) {
	if (uct_child[i].flag == false) {
	  int pos = uct_child[i].pos;
	  dynamic_parameter = uct_owner[owner_index[pos]] + uct_criticality[criticality_index[pos]];
	  if (uct_child[i].rate + dynamic_parameter > max_rate) {
	    max_index = i;
	    max_rate = uct_child[i].rate + dynamic_parameter;
	  }
	}
      }
      if (max_index != -1) {
	uct_child[max_index].flag = true;
      }
      uct_node[current].width++;
    }
  }

  double max_value = -1;
  int max_child = 0;

  const double p_p = (double)uct_node[current].win / uct_node[current].move_count;
  const double p_v = (double)uct_node[current].value_win / (uct_node[current].value_move_count + .01);
  const double scale = std::max(0.2, std::min(1.0, 1.0 - (game->moves - 200) / 50.0)) * value_scale;

  int start_child = 0;
  if (!early_pass && current == current_root && child_num > 1) {
    if (uct_child[0].move_count > uct_node[current].move_count * pass_po_limit) {
      start_child = 1;
    }
  }

  int max_move_count = 0;
  int max_move_child = 0;

  // UCB値最大の手を求める
  for (int i = start_child; i < child_num; i++) {
    if (uct_child[i].flag || uct_child[i].open) {
      //double p2 = -1;
      double value_win = 0;
      double value_move_count = 0;

#if 1
      if (uct_child[i].index >= 0 && i != 0) {
	auto node = &uct_node[uct_child[i].index];
	if (node->value_move_count > 0) {
	  //p2 = 1 - (double)node->value_win / node->value_move_count;
	  value_win = node->value_win;
	  value_move_count = node->value_move_count;
	  value_win = value_move_count - value_win;
	}
	//cerr << "VA:" << (value_win / value_move_count) << " VS:" << uct_child[i].value << endl;
      }
      if (value_move_count == 0 && uct_child[i].value >= 0) {
	value_move_count = 1;
	value_win = uct_child[i].value;
      }
#endif
      double win = uct_child[i].win;
      double move_count = uct_child[i].move_count;
      double ucb_value, lcb_value;
      double p;

      if (evaled) {
	if (debug && move_count > 0) {
	   cerr << uct_node[current].move_count << ".";
	   cerr << setw(3) << FormatMove(uct_child[i].pos);
	   cerr << ": move " << setw(5) << move_count << " policy "
	    << setw(10) << (uct_child[i].nnrate  * 100) << " ";
	}
	if (move_count == 0) {
	  p = p_p * (1 - scale) + p_v * scale;
	} else {
	  double p0 = win / move_count;
	  if (value_move_count > 0) {
	    double p1 = value_win / value_move_count;
	    //p = (uct_child[i].win + value_win) / (uct_child[i].move_count + value_move_count);
	    p = p0 * (1 - scale)  + p1 * scale;
	    //p = (p0 + p1) / 2;
	    //if (current == current_root) cerr << i << ":" << p0 << " " << p1 << " => " << p << endl;
	    if (debug) {
		cerr << "DP:" << setw(10) << (p0 * 100) << " DV:" << setw(10) << (p1 * 100) << " => " << setw(10) << (p * 100)
		<< " " << p_v << " V:" << (value_win / value_move_count);
		cerr << " LM:" << scale << " ";
	    }
	  } else {
	    p = p0 * (1 - scale) + p_v * scale;
	  }
	}

	double u = sqrt(sum) / (1 + uct_child[i].move_count);
	double rate = uct_child[i].nnrate;
	ucb_value = p + c_puct * u * rate;
        lcb_value = p - c_puct * u * rate;

	if (debug && move_count > 0) {
	  cerr << " P:" << p << " UCB:" << ucb_value << endl;
	}
      } else {
	if (uct_child[i].move_count == 0) {
	  ucb_value = FPU;
	  lcb_value = FPU;
	} else {
	  double div, v;
	  // UCB1-TUNED value
	  p = (double) uct_child[i].win / uct_child[i].move_count;
	  //if (p2 >= 0) p = (p * 9 + p2) / 10;
	  div = log(sum) / uct_child[i].move_count;
	  v = p - p * p + sqrt(2.0 * div);
	  ucb_value = p + sqrt(div * ((0.25 < v) ? 0.25 : v));
	  lcb_value = p - sqrt(div * ((0.25 < v) ? 0.25 : v));

	  // UCB Bonus
	  ucb_value += ucb_bonus_weight * uct_child[i].rate;
	  lcb_value += ucb_bonus_weight * uct_child[i].rate;
	}
      }

      child_ucb[i] = ucb_value;
      child_lcb[i] = lcb_value;

      if (ucb_value > max_value) {
	max_value = ucb_value;
	max_child = i;
      }
      if (uct_child[i].move_count > max_move_count) {
	max_move_count = uct_child[i].move_count;
	max_move_child = i;
      }
    }
  }

  if (current == current_root && max_child == max_move_child) {
    double next_ucb = child_lcb[max_child];
    int next_child = max_child;
    for (int i = 0; i < child_num; i++) {
      if (max_child == i)
	continue;
      if (uct_child[i].flag || uct_child[i].open) {
	if (child_ucb[i] > next_ucb) {
	  next_ucb = child_ucb[i];
	  next_child = i;
	}
      }
    }
    if (max_child != next_child
	&& uct_child[max_child].move_count > uct_child[next_child].move_count * 1.2) {
      //cerr << "Replace " << FormatMove(uct_child[max_child].pos) << " -> " << FormatMove(uct_child[next_child].pos) << endl;
      max_child = next_child;
    }
  }

  /*
  static ray_clock::time_point previous_time = ray_clock::now();
  static mutex mutex_log;
  if (current == current_root && sum > 0) {
    mutex_log.lock();
    if (GetSpendTime(previous_time) > 1.0) {
      for (int i = 0; i < child_num; i++) {
	if (i > 0 && !uct_child[i].flag && !uct_child[i].open)
	  continue;
	double win = uct_child[i].win;
	double move_count = uct_child[i].move_count;
	double p0 = win / move_count;

	cerr << "|" << setw(4) << FormatMove(uct_child[i].pos);
	cerr << "|" << setw(5) << (int) move_count;
	auto precision = cerr.precision();
	cerr.precision(4);
	cerr << "|" << setw(10) << fixed << (p0 * 100);
	cerr << "|" << setw(10) << fixed << (child_ucb[i] * 100);
	cerr << "|" << setw(10) << fixed << (child_lcb[i] * 100);
	cerr << endl;
	cerr.precision(precision);
      }
      previous_time = ray_clock::now();
    }
    mutex_log.unlock();
  }
  */

  return max_child;
}


///////////////////////////////////////////////////////////
//  OwnerやCriiticalityを計算するための情報を記録する関数  //
///////////////////////////////////////////////////////////
static void
Statistic( game_info_t *game, int winner )
{
  const char *board = game->board;

  for (int i = 0; i < pure_board_max; i++) {
    const int pos = onboard_pos[i];
    int color = board[pos];

    if (color == S_EMPTY) color = territory[Pat3(game->pat, pos)];

    std::atomic_fetch_add(&statistic[pos].colors[color], 1);
    if (color == winner) {
      std::atomic_fetch_add(&statistic[pos].colors[0], 1);
    }
  }
}


///////////////////////////////
//  各ノードの統計情報の更新  //
///////////////////////////////
static void
UpdateNodeStatistic( game_info_t *game, int winner, statistic_t *node_statistic )
{
  const char *board = game->board;

  for (int i = 0; i < pure_board_max; i++) {
    const int pos = onboard_pos[i];
    int color = board[pos];

    if (color == S_EMPTY) color = territory[Pat3(game->pat, pos)];

    std::atomic_fetch_add(&node_statistic[pos].colors[color], 1);
    if (color == winner) {
      std::atomic_fetch_add(&node_statistic[pos].colors[0], 1);
    }
  }
}


//////////////////////////////////
//  各ノードのCriticalityの計算  //
//////////////////////////////////
static void
CalculateCriticalityIndex( uct_node_t *node, statistic_t *node_statistic, int color, int *index )
{
  const int other = FLIP_COLOR(color);
  const int count = node->move_count;
  const int child_num = node->child_num;
  const double win = (double)node->win / node->move_count;
  const double lose = 1.0 - win;
  double tmp;

  index[0] = 0;

  for (int i = 1; i < child_num; i++) {
    const int pos = node->child[i].pos;

    tmp = ((double)node_statistic[pos].colors[0] / count) -
      ((((double)node_statistic[pos].colors[color] / count) * win)
       + (((double)node_statistic[pos].colors[other] / count) * lose));
    if (tmp < 0) tmp = 0;
    index[i] = (int)(tmp * 40);
    if (index[i] > criticality_max - 1) index[i] = criticality_max - 1;
  }
}

////////////////////////////////////
//  Criticalityの計算をする関数   //
////////////////////////////////////
static void
CalculateCriticality( int color )
{
  const int other = FLIP_COLOR(color);
  const double win = (double)uct_node[current_root].win / uct_node[current_root].move_count;
  const double lose = 1.0 - win;
  double tmp;

  for (int i = 0; i < pure_board_max; i++) {
    const int pos = onboard_pos[i];

    tmp = ((float)statistic[pos].colors[0] / po_info.count) -
      ((((float)statistic[pos].colors[color] / po_info.count)*win)
       + (((float)statistic[pos].colors[other] / po_info.count)*lose));
    criticality[pos] = tmp;
    if (tmp < 0) tmp = 0;
    criticality_index[pos] = (int)(tmp * 40);
    if (criticality_index[pos] > criticality_max - 1) criticality_index[pos] = criticality_max - 1;
  }
}


//////////////////////////////
//  Ownerの計算をする関数   //
//////////////////////////////
static void
CalculateOwnerIndex( uct_node_t *node, statistic_t *node_statistic, int color, int *index )
{
  const int count = node->move_count;
  const int child_num = node->child_num;

  index[0] = 0;

  for (int i = 1; i < child_num; i++){
    const int pos = node->child[i].pos;
    index[i] = (int)((double)node_statistic[pos].colors[color] * 10.0 / count + 0.5);
    if (index[i] > OWNER_MAX - 1) index[i] = OWNER_MAX - 1;
    if (index[i] < 0)             index[i] = 0;
  }
}


//////////////////////////////
//  Ownerの計算をする関数   //
//////////////////////////////
static void
CalculateOwner( int color, int count )
{
  for (int i = 0; i < pure_board_max; i++){
    const int pos = onboard_pos[i];
    owner_index[pos] = (int)((double)statistic[pos].colors[color] * 10.0 / count + 0.5);
    if (owner_index[pos] > OWNER_MAX - 1) owner_index[pos] = OWNER_MAX - 1;
    if (owner_index[pos] < 0)             owner_index[pos] = 0;
  }
}


/////////////////////////////////
//  次のプレイアウト回数の設定  //
/////////////////////////////////
static void
CalculateNextPlayouts( game_info_t *game, int color, double best_wp, double finish_time )
{
  double po_per_sec;

  if (finish_time != 0.0) {
    po_per_sec = po_info.count / finish_time;
  } else {
    po_per_sec = PLAYOUT_SPEED * threads;
  }

  // 次の探索の時の探索回数を求める
  if (mode == CONST_TIME_MODE) {
    if (best_wp > 0.90) {
      po_info.num = (int)(po_info.count / finish_time * const_thinking_time / 2);
    } else {
      po_info.num = (int)(po_info.count / finish_time * const_thinking_time);
    }
  } else if (mode == TIME_SETTING_MODE ||
	     mode == TIME_SETTING_WITH_BYOYOMI_MODE) {
    remaining_time[color] -= finish_time;
    if (pure_board_size < 11) {
      time_limit = remaining_time[color] / TIME_RATE_9;
    } else if (pure_board_size < 16) {
      time_limit = remaining_time[color] / (TIME_C_13 + ((TIME_MAXPLY_13 - (game->moves + 1) > 0) ? TIME_MAXPLY_13 - (game->moves + 1) : 0));
    } else {
      time_limit = remaining_time[color] / (TIME_C_19 + ((TIME_MAXPLY_19 - (game->moves + 1) > 0) ? TIME_MAXPLY_19 - (game->moves + 1) : 0));
    }
    if (mode == TIME_SETTING_WITH_BYOYOMI_MODE &&
	time_limit < (const_thinking_time * 0.5)) {
      time_limit = const_thinking_time * 0.5;
    }
    po_info.num = (int)(po_per_sec * time_limit);
  }
}


/////////////////////////////////////
//  UCTアルゴリズムによる局面解析  //
/////////////////////////////////////
int
UctAnalyze( game_info_t *game, int color )
{
  vector<unique_ptr<std::thread>> handle;

  // 探索情報をクリア
  memset(statistic, 0, sizeof(statistic_t) * board_max);
  fill_n(criticality_index, board_max, 0);
  for (int i = 0; i < board_max; i++) {
    criticality[i] = 0.0;
  }
  po_info.count = 0;

  ClearUctHash();

  current_root = ExpandRoot(game, color);

  bool org_use_nn = use_nn;
  use_nn = false;

  po_info.halt = 10000;

  t_arg.resize(threads);
  running = true;
  for (int i = 0; i < threads; i++) {
    t_arg[i].thread_id = i;
    t_arg[i].game = game;
    t_arg[i].color = color;
    handle.push_back(make_unique<thread>(ParallelUctSearch, &t_arg[i]));
  }

  for (auto &t : handle) {
    t->join();
  }
  handle.clear();

  use_nn = org_use_nn;

  int black = 0, white = 0;

  for (int y = board_start; y <= board_end; y++) {
    for (int x = board_start; x <= board_end; x++) {
      const int pos = POS(x, y);
      const double ownership_value = (double)statistic[pos].colors[S_BLACK] / uct_node[current_root].move_count;
      if (ownership_value > 0.5) {
	black++;
      } else {
	white++;
      }
    }
  }

  PrintOwner(&uct_node[current_root], color, owner);

  return black - white;
}


/////////////////////////
//  Ownerをコピーする  //
/////////////////////////
void
OwnerCopy( int *dest )
{
  for (int i = 0; i < pure_board_max; i++) {
    const int pos = onboard_pos[i];
    dest[pos] = (int)((double)uct_node[current_root].statistic[pos].colors[my_color] / uct_node[current_root].move_count * 100);
  }
}


///////////////////////////////
//  Criticalityをコピーする  //
///////////////////////////////
void
CopyCriticality( double *dest )
{
  for (int i = 0; i < pure_board_max; i++) {
    const int pos = onboard_pos[i];
    dest[pos] = criticality[pos];
  }
}

void
CopyStatistic( statistic_t *dest )
{
  memcpy(dest, statistic, sizeof(statistic_t)* BOARD_MAX);
}


////////////////////////////////////////////////////////
//  UCTアルゴリズムによる着手生成(KGS Clean Up Mode)  //
////////////////////////////////////////////////////////
int
UctSearchGenmoveCleanUp( game_info_t *game, int color )
{
  int pos, select_index, max_count, count;
  double finish_time, wp;
  child_node_t *uct_child;
  vector<unique_ptr<std::thread>> handle;

  memset(statistic, 0, sizeof(statistic_t)* board_max);
  fill_n(criticality_index, board_max, 0);
  for (int i = 0; i < board_max; i++) {
    criticality[i] = 0.0;
  }

  begin_time = ray_clock::now();

  po_info.count = 0;

  current_root = ExpandRoot(game, color);

  if (uct_node[current_root].child_num <= 1) {
    return PASS;
  }

  for (int i = 0; i < pure_board_max; i++) {
    pos = onboard_pos[i];
    owner[pos] = 50.0;
  }

  po_info.halt = po_info.num;

  bool org_use_nn = use_nn;
  use_nn = false;

  DynamicKomi(game, &uct_node[current_root], color);

  t_arg.reserve(threads);
  running = true;
  for (int i = 0; i < threads; i++) {
    t_arg[i].thread_id = i;
    t_arg[i].game = game;
    t_arg[i].color = color;
    handle.push_back(make_unique<thread>(ParallelUctSearch, &t_arg[i]));
  }

  for (auto &t : handle) {
    t->join();
  }
  handle.clear();

  use_nn = org_use_nn;

  uct_child = uct_node[current_root].child;

  select_index = 0;
  max_count = uct_child[0].move_count;

  for (int i = 0; i < uct_node[current_root].child_num; i++){
    if (uct_child[i].move_count > max_count) {
      select_index = i;
      max_count = uct_child[i].move_count;
    }
  }

  finish_time = GetSpendTime(begin_time);

  wp = (double)uct_node[current_root].win / uct_node[current_root].move_count;

  PrintPlayoutInformation(&uct_node[current_root], &po_info, finish_time, 0);
  PrintOwner(&uct_node[current_root], color, owner);

  pos = uct_child[select_index].pos;

  PrintBestSequence(game, uct_node, current_root, color);

  CalculateNextPlayouts(game, color, wp, finish_time);

  count = 0;

  for (int i = 0; i < pure_board_max; i++) {
    pos = onboard_pos[i];

    if (owner[pos] >= 5 || owner[pos] <= 95) {
      candidates[pos] = true;
      count++;
    } else {
      candidates[pos] = false;
    }
  }

  if (count == 0) {
    pos = PASS;
  } else {
    pos = uct_child[select_index].pos;
  }

  if ((double)uct_child[select_index].win / uct_child[select_index].move_count < resign_threshold) {
    return PASS;
  } else {
    return pos;
  }
}

///////////////////////////////////
//  子ノードのインデックスの収集  //
///////////////////////////////////
static void
CorrectDescendentNodes(vector<int> &indexes, int index)
{
  child_node_t *uct_child = uct_node[index].child;
  const int child_num = uct_node[index].child_num;

  indexes.push_back(index);

  for (int i = 0; i < child_num; i++) {
    if (uct_child[i].index != NOT_EXPANDED) {
      CorrectDescendentNodes(indexes, uct_child[i].index);
    }
  }
}

extern char uct_params_path[1024];

static CNTK::DeviceDescriptor
GetDevice()
{
  if (device_id == -1)
    return CNTK::DeviceDescriptor::CPUDevice();
  if (device_id == -2)
    return CNTK::DeviceDescriptor::UseDefaultDevice();
  return CNTK::DeviceDescriptor::GPUDevice(device_id);
}

void
ReadWeights()
{
  wchar_t name[1024];
  mbstate_t ps;
  memset(&ps, 0, sizeof(ps));
  const char * src = uct_params_path;
  mbsrtowcs(name, &src, 1024, &ps);
  wstring path = name;

  cerr << "Init CNTK" << endl;

  auto device = GetDevice();

  wstring policy_name = path;
  policy_name += L"/model2.bin";
  nn_policy = CNTK::Function::Load(policy_name, device);

  wstring value_name = path;
  value_name += L"/model3.bin";
  nn_value = CNTK::Function::Load(value_name, device);

  if (!nn_policy || !nn_value)
  {
    cerr << "Get EvalModel failed\n";
    abort();
  }

#if 0
  wcerr << L"***POLICY" << endl;
  for (auto var : nn_policy->Inputs()) {
    wcerr << var.AsString() << endl;
  }
  for (auto var : nn_policy->Outputs()) {
    wcerr << var.AsString() << endl;
  }
  wcerr << L"***VALUE" << endl;
  for (auto var : nn_value->Inputs()) {
    wcerr << var.AsString() << endl;
  }
  for (auto var : nn_value->Outputs()) {
    wcerr << var.AsString() << endl;
  }
#endif

  cerr << "ok" << endl;
}


bool
GetVariableByName(vector<CNTK::Variable> variableLists, wstring varName, CNTK::Variable& var)
{
  for (vector<CNTK::Variable>::iterator it = variableLists.begin(); it != variableLists.end(); ++it)
  {
    if (it->Name().compare(varName) == 0)
    {
      var = *it;
      return true;
    }
  }
  return false;
}

inline bool
GetInputVariableByName(CNTK::FunctionPtr evalFunc, wstring varName, CNTK::Variable& var)
{
  return GetVariableByName(evalFunc->Arguments(), varName, var);
}

inline bool
GetOutputVaraiableByName(CNTK::FunctionPtr evalFunc, wstring varName, CNTK::Variable& var)
{
  return GetVariableByName(evalFunc->Outputs(), varName, var);
}

void
EvalPolicy(
  const std::vector<std::shared_ptr<policy_eval_req>>& requests,
  std::vector<float>& data_basic, std::vector<float>& data_features, std::vector<float>& data_history,
  std::vector<float>& data_color, std::vector<float>& data_komi)
{
  if (requests.size() == 0)
    return;

  auto device = GetDevice();

  CNTK::Variable var_basic, var_features, var_history, var_color, var_komi;
  GetInputVariableByName(nn_policy, L"basic", var_basic);
  GetInputVariableByName(nn_policy, L"features", var_features);
  GetInputVariableByName(nn_policy, L"history", var_history);
  GetInputVariableByName(nn_policy, L"color", var_color);
  GetInputVariableByName(nn_policy, L"komi", var_komi);

  CNTK::Variable var_ol;
  GetOutputVaraiableByName(nn_policy, L"ol", var_ol);

  size_t num_req = requests.size();

  CNTK::NDShape shape_basic = var_basic.Shape().AppendShape({ 1, num_req });
  CNTK::ValuePtr value_basic = CNTK::MakeSharedObject<CNTK::Value>(CNTK::MakeSharedObject<CNTK::NDArrayView>(shape_basic, data_basic, true));
  CNTK::NDShape shape_features = var_features.Shape().AppendShape({ 1, num_req });
  CNTK::ValuePtr value_features = CNTK::MakeSharedObject<CNTK::Value>(CNTK::MakeSharedObject<CNTK::NDArrayView>(shape_features, data_features, true));
  CNTK::NDShape shape_history = var_history.Shape().AppendShape({ 1, num_req });
  CNTK::ValuePtr value_history = CNTK::MakeSharedObject<CNTK::Value>(CNTK::MakeSharedObject<CNTK::NDArrayView>(shape_history, data_history, true));
  CNTK::NDShape shape_color = var_color.Shape().AppendShape({ 1, num_req });
  CNTK::ValuePtr value_color = CNTK::MakeSharedObject<CNTK::Value>(CNTK::MakeSharedObject<CNTK::NDArrayView>(shape_color, data_color, true));
  //CNTK::NDShape shape_komi = var_komi.Shape().AppendShape({ 1, num_req });
  //CNTK::ValuePtr value_komi = CNTK::MakeSharedObject<CNTK::Value>(CNTK::MakeSharedObject<CNTK::NDArrayView>(shape_komi, data_komi, true));

  CNTK::ValuePtr value_ol;

  std::unordered_map<CNTK::Variable, CNTK::ValuePtr> inputs = {
    { var_basic, value_basic },
    { var_features, value_features },
    { var_history, value_history },
    //{ var_color, value_color },
    //{ var_komi, value_komi },
  };
  std::unordered_map<CNTK::Variable, CNTK::ValuePtr> outputs = { { var_ol, value_ol } };

  try {
    nn_policy->Forward(inputs, outputs, device);
  } catch (const std::exception& err) {
    fprintf(stderr, "Evaluation failed. EXCEPTION occurred: %s\n", err.what());
    abort();
  } catch (...) {
    fprintf(stderr, "Evaluation failed. Unknown ERROR occurred.\n");
    abort();
  }

  value_ol = outputs[var_ol];
  CNTK::NDShape shape_ol = var_ol.Shape().AppendShape({ 1, num_req });
  vector<float> moves(shape_ol.TotalSize());
  CNTK::NDArrayViewPtr cpu_moves = CNTK::MakeSharedObject<CNTK::NDArrayView>(shape_ol, moves, false);
  cpu_moves->CopyFrom(*value_ol->Data());

  if (moves.size() != pure_board_max * num_req) {
    cerr << "Eval move error " << moves.size() << endl;
    return;
  }

  for (int j = 0; j < requests.size(); j++) {
    const auto req = requests[j];
    const int index = req->index;
    const int child_num = uct_node[index].child_num;
    child_node_t *uct_child = uct_node[index].child;
    const int ofs = pure_board_max * j;

    LOCK_NODE(index);

    int depth = req->depth;
#if 0
    if (index == current_root) {
      for (int i = 0; i < pure_board_max; i++) {
	int x = i % pure_board_size;
	int y = i / pure_board_size;
	owner_nn[POS(x + OB_SIZE, y + OB_SIZE)] = ownern[i + ofs];
      }
    }
#endif

    for (int i = 1; i < child_num; i++) {
      int pos = RevTransformMove(uct_child[i].pos, req->trans);

      int x = X(pos) - OB_SIZE;
      int y = Y(pos) - OB_SIZE;
      int n = x + y * pure_board_size;
      double score = moves[n + ofs];
      //if (depth == 1) cerr << "RAW POLICY " << uct_child[i].pos << " " << req->trans << " " << FormatMove(pos) << " " << x << "," << y << " " << ofs << " -> " << score << endl;
      if (uct_child[i].ladder) {
        score -= 4; // ~= 1.83%
      }

      uct_child[i].nnrate0 = score;
    }

    UpdatePolicyRate(index);
    uct_node[index].evaled = true;

    UNLOCK_NODE(index);
  }
  eval_count_policy += requests.size();
}


void
EvalValue(
  const std::vector<std::shared_ptr<value_eval_req>>& requests,
  std::vector<float>& data_basic, std::vector<float>& data_features, std::vector<float>& data_history,
  std::vector<float>& data_color, std::vector<float>& data_komi, std::vector<float>& data_safety)
{
  if (requests.size() == 0)
    return;

  auto device = GetDevice();

  CNTK::Variable var_basic, var_features, var_history, var_color, var_komi, var_safety;
  GetInputVariableByName(nn_value, L"basic", var_basic);
  GetInputVariableByName(nn_value, L"features", var_features);
  GetInputVariableByName(nn_value, L"history", var_history);
  GetInputVariableByName(nn_value, L"color", var_color);
  GetInputVariableByName(nn_value, L"komi", var_komi);
  GetInputVariableByName(nn_value, L"safety", var_safety);

  CNTK::Variable var_p;
  GetOutputVaraiableByName(nn_value, L"p", var_p);

  size_t num_req = requests.size();

  CNTK::NDShape shape_basic = var_basic.Shape().AppendShape({ 1, num_req });
  CNTK::ValuePtr value_basic = CNTK::MakeSharedObject<CNTK::Value>(CNTK::MakeSharedObject<CNTK::NDArrayView>(shape_basic, data_basic, true));
  CNTK::NDShape shape_features = var_features.Shape().AppendShape({ 1, num_req });
  CNTK::ValuePtr value_features = CNTK::MakeSharedObject<CNTK::Value>(CNTK::MakeSharedObject<CNTK::NDArrayView>(shape_features, data_features, true));
  CNTK::NDShape shape_history = var_history.Shape().AppendShape({ 1, num_req });
  CNTK::ValuePtr value_history = CNTK::MakeSharedObject<CNTK::Value>(CNTK::MakeSharedObject<CNTK::NDArrayView>(shape_history, data_history, true));
  CNTK::NDShape shape_color = var_color.Shape().AppendShape({ 1, num_req });
  CNTK::ValuePtr value_color = CNTK::MakeSharedObject<CNTK::Value>(CNTK::MakeSharedObject<CNTK::NDArrayView>(shape_color, data_color, true));
  CNTK::NDShape shape_komi = var_komi.Shape().AppendShape({ 1, num_req });
  CNTK::ValuePtr value_komi = CNTK::MakeSharedObject<CNTK::Value>(CNTK::MakeSharedObject<CNTK::NDArrayView>(shape_komi, data_komi, true));
  CNTK::NDShape shape_safety = var_safety.Shape().AppendShape({ 1, num_req });
  CNTK::ValuePtr value_safety = CNTK::MakeSharedObject<CNTK::Value>(CNTK::MakeSharedObject<CNTK::NDArrayView>(shape_safety, data_safety, true));

  CNTK::ValuePtr value_p;

  std::unordered_map<CNTK::Variable, CNTK::ValuePtr> inputs = {
    { var_basic, value_basic },
    { var_features, value_features },
    { var_history, value_history },
    { var_color, value_color },
    { var_komi, value_komi },
    { var_safety, value_safety },
  };
  std::unordered_map<CNTK::Variable, CNTK::ValuePtr> outputs = { { var_p, value_p } };

  try {
    nn_value->Forward(inputs, outputs, device);
  } catch (const std::exception& err) {
    fprintf(stderr, "Evaluation failed. EXCEPTION occurred: %s\n", err.what());
    abort();
  } catch (...) {
    fprintf(stderr, "Evaluation failed. Unknown ERROR occurred.\n");
    abort();
  }

  value_p = outputs[var_p];
  CNTK::NDShape shape_p = var_p.Shape().AppendShape({ 1, num_req });
  vector<float> win(shape_p.TotalSize());
  CNTK::NDArrayViewPtr cpu_p = CNTK::MakeSharedObject<CNTK::NDArrayView>(shape_p, win, false);
  cpu_p->CopyFrom(*value_p->Data());

  if (win.size() != requests.size()) {
    cerr << "Eval win error " << win.size() << endl;
    return;
  }
  //cerr << "Eval " << indices.size() << " " << path.size() << endl;
  for (int j = 0; j < requests.size(); j++) {
    auto req = requests[j];

    double p = ((double)win[j] + 1) / 2;
    if (p < 0)
      p = 0;
    if (p > 1)
      p = 1;
    //cerr << "#" << index << "  " << sum << endl;

    double value = 1 - p;// color[j] == S_BLACK ? p : 1 - p;

    req->uct_child->value = value;
    for (int i = req->path.size() - 1; i >= 0; i--) {
      int current = req->path[i];
      if (current < 0)
	break;

      atomic_fetch_add(&uct_node[current].value_move_count, 1);
      atomic_fetch_add(&uct_node[current].value_win, value);
      value = 1 - value;
    }
  }
  eval_count_value += requests.size();
}

void EvalNode() {
  std::vector<float> eval_input_data_basic;
  std::vector<float> eval_input_data_features;
  std::vector<float> eval_input_data_history;
  std::vector<float> eval_input_data_color;
  std::vector<float> eval_input_data_komi;
  std::vector<float> eval_input_data_safety;

  int num_eval = 0;
  bool allow_skip = (!reuse_subtree && !ponder) || time_limit <= 1.0;

  while (true) {
    mutex_queue.lock();
    if (!running
      && (allow_skip || (eval_policy_queue.empty() && eval_value_queue.empty()))) {
      mutex_queue.unlock();
      cerr << "Eval " << num_eval << endl;
      break;
    }

    if (eval_policy_queue.empty() && eval_value_queue.empty()) {
      value_evaluation_threshold = max(0.0, value_evaluation_threshold - 0.01);
      mutex_queue.unlock();
      this_thread::sleep_for(chrono::milliseconds(1));
      //cerr << "EMPTY QUEUE" << endl;
      continue;
    }

    if (eval_policy_queue.size() == 0) {
    } else {
      std::vector<std::shared_ptr<policy_eval_req>> requests;

      for (int i = 0; i < policy_batch_size && !eval_policy_queue.empty(); i++) {
        auto req = eval_policy_queue.front();
        requests.push_back(req);
        eval_policy_queue.pop();
      }
      mutex_queue.unlock();

      eval_input_data_basic.resize(0);
      eval_input_data_features.resize(0);
      eval_input_data_history.resize(0);
      eval_input_data_color.resize(0);
      eval_input_data_komi.resize(0);
      for (auto& req : requests) {
        std::copy(req->data_basic.begin(), req->data_basic.end(), std::back_inserter(eval_input_data_basic));
        std::copy(req->data_features.begin(), req->data_features.end(), std::back_inserter(eval_input_data_features));
        std::copy(req->data_history.begin(), req->data_history.end(), std::back_inserter(eval_input_data_history));
        eval_input_data_color.push_back(req->color - 1);
        eval_input_data_komi.push_back(komi[0]);
      }
      num_eval += requests.size();
      EvalPolicy(requests, eval_input_data_basic, eval_input_data_features, eval_input_data_history, eval_input_data_color, eval_input_data_komi);
      mutex_queue.lock();
    }

    if (eval_value_queue.size() == 0) {
      mutex_queue.unlock();
    } else {
      std::vector<std::shared_ptr<value_eval_req>> requests;

      for (int i = 0; i < value_batch_size && !eval_value_queue.empty(); i++) {
        auto req = eval_value_queue.front();
        requests.push_back(req);
        eval_value_queue.pop();
      }
      mutex_queue.unlock();

      eval_input_data_basic.resize(0);
      eval_input_data_features.resize(0);
      eval_input_data_history.resize(0);
      eval_input_data_color.resize(0);
      eval_input_data_komi.resize(0);
      for (auto& req : requests) {
        std::copy(req->data_basic.begin(), req->data_basic.end(), std::back_inserter(eval_input_data_basic));
        std::copy(req->data_features.begin(), req->data_features.end(), std::back_inserter(eval_input_data_features));
        std::copy(req->data_history.begin(), req->data_history.end(), std::back_inserter(eval_input_data_history));
        eval_input_data_color.push_back(req->color - 1);
        eval_input_data_komi.push_back(komi[0]);
      }
      eval_input_data_safety.resize(requests.size() * pure_board_max * 8);
      num_eval += requests.size();
      EvalValue(requests, eval_input_data_basic, eval_input_data_features, eval_input_data_history, eval_input_data_color, eval_input_data_komi, eval_input_data_safety);
    }
  }
}
