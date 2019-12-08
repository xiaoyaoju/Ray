#include "Train.h"
#include "UctSearch.h"
//#include "SgfExtractor.h"
#include "DynamicKomi.h"
#include "Rating.h"
#include "Message.h"
#include "MoveCache.h"
#include "Simulation.h"
#include "Utility.h"

#include <filesystem>
#include <iostream>
#include <fstream>
#include <mutex>
#include <queue>
#include <random>
#include <thread>
#include <vector>

#include <boost/iostreams/copy.hpp>
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/interprocess/streams/vectorstream.hpp>

using namespace std;
using namespace CNTK;

#ifdef  _WIN32
namespace fs = std::experimental::filesystem;
#endif

struct LZ_record_t {
  string filename;
  vector<vector<size_t>> pos;
};

const size_t num_games_per_file = (size_t)10;

static string kifu_dir = "kifu";
static string test_kifu_dir = "";
static bool use_easy_state = false;
static bool use_smooth_value = false;
static std::mt19937_64 mt;
static mutex extract_mutex;

void
SetKifuDirectory(const std::string dir)
{
  kifu_dir = dir;
}

void
SetTestKifuDirectory(const std::string dir)
{
  test_kifu_dir = dir;
}

static void Inflate(const string &filename, vector<char> &decompressed)
{
  ifstream fin(filename, ios_base::in | ios_base::binary);

  boost::iostreams::filtering_ostream os;

  os.push(boost::iostreams::gzip_decompressor());
  os.push(boost::iostreams::back_inserter(decompressed));

  boost::iostreams::copy(fin, os);
}

inline void PrintTrainingProgress(const TrainerPtr trainer, size_t minibatchIdx, size_t outputFrequencyInMinibatches)
{
  if ((minibatchIdx % outputFrequencyInMinibatches) == 0 && trainer->PreviousMinibatchSampleCount() != 0)
  {
    double trainLossValue = trainer->PreviousMinibatchLossAverage();
    double evaluationValue = trainer->PreviousMinibatchEvaluationAverage();
    fprintf(stderr, "Minibatch %d: CrossEntropy loss = %.8g, Evaluation criterion = %.8g\n", (int)minibatchIdx, trainLossValue, evaluationValue);
    fflush(stderr);
    //double trainLossValue = trainer->PreviousMinibatchLossAverage();
    //printf("Minibatch %d: CrossEntropy loss = %.8g\n", (int)minibatchIdx, trainLossValue);
  }
}

void ReadLzPlane(const string& plane, int color, game_info_t* game)
{
  for (auto n = size_t{ 0 }, bit = size_t{ 0 }; n < plane.size(); n++, bit += 4) {
    char c = plane[n];
    int hexbyte;
    if (c >= '0' && c <= '9') {
      hexbyte = c - '0';
    }
    else if (c >= 'a' && c <= 'f') {
      hexbyte = 10 + c - 'a';
    }
    else {
      cerr << "illegal input" << endl;
      return;
    }
    if (bit < pure_board_max
      && (hexbyte & 8) != 0 && game->board[onboard_pos[bit]] == S_EMPTY)
      PutStone(game, onboard_pos[bit], color);
    if (bit + 1 < pure_board_max
      && (hexbyte & 4) != 0 && game->board[onboard_pos[bit + 1]] == S_EMPTY)
      PutStone(game, onboard_pos[bit + 1], color);
    if (bit + 2 < pure_board_max
      && (hexbyte & 2) != 0 && game->board[onboard_pos[bit + 2]] == S_EMPTY)
      PutStone(game, onboard_pos[bit + 2], color);
    if (bit + 3 < pure_board_max
      && (hexbyte & 1) != 0 && game->board[onboard_pos[bit + 3]] == S_EMPTY)
      PutStone(game, onboard_pos[bit + 3], color);
  }
}

bool ReadLzTrainingData(istream& in, game_info_t* game, int& color, float* prob, int& win)
{
  ClearBoard(game);
  string dummy;
  string line[16];
  for (int i = 0; i < 16; i++) {
    getline(in, line[i]);
    if (line[i].size() < 10) {
      if (i == 0) {
        if (in.eof())
          return false;
      }
      cerr
        << "illegal plane: " << line[i]
        << endl;
    }
  }
  if (in.eof())
    return false;
  int c;
  in >> c;
  if (c == 0)
    color = S_BLACK;
  else
    color = S_WHITE;

  getline(in, dummy);
  if (dummy.length() != 0)
    abort();
  for (int i = 7; i >= 0; i--) {
    int moves0 = game->moves;
    ReadLzPlane(line[color == S_BLACK ? i : 8 + i], S_BLACK, game);
    ReadLzPlane(line[color == S_BLACK ? 8 + i : i], S_WHITE, game);
    if (game->moves > 1 && game->moves == moves0) {
      PutStone(game, PASS, FLIP_COLOR(game->record[game->moves - 1].color));
    }
    // if (win == 100) PrintBoard(game);
  }

#if 0
  for (int i = 0; i < pure_board_max + 1; i++) {
    in >> prob[i];
    if (prob[i] > 0.01) {
      // cerr << FormatMove(i == pure_board_max ? PASS : onboard_pos[i]) << ":" << prob[i] << endl;
    }
  }
#else
  string prob_line;
  getline(in, prob_line);

  const char* ptr = prob_line.c_str();
  for (int i = 0; i < pure_board_max + 1; i++) {
    if (*ptr == '\0') {
      cerr << "Illegal data" << endl;
      abort();
    }
    prob[i] = atof(ptr);
    while (*ptr != '\0' && *ptr != ' ')
      ptr++;
    ptr++;
    if (prob[i] > 0.01) {
      // cerr << FormatMove(i == pure_board_max ? PASS : onboard_pos[i]) << ":" << prob[i] << endl;
    }
  }
#endif
  in >> c;
  if (c > 0)
    win = color;
  else
    win = FLIP_COLOR(color);

  getline(in, dummy);
  if (dummy.length() != 0)
    abort();
  return true;
}

void
ExtractKifu(LZ_record_t* rec)
{
  auto begin_time = ray_clock::now();

  const auto& filename = rec->filename;
  ifstream index_in(filename + ".idx");
  if (!index_in.fail()) {
    while (!index_in.eof()) {
      string line;
      getline(index_in, line);

      if (line.size() == 0)
        break;
      rec->pos.push_back({});

      stringstream in(line);
      while (!in.eof()) {
        size_t pos;
        in >> pos;
        rec->pos[rec->pos.size() - 1].push_back(pos);
      }
    }
    auto finish_time = GetSpendTime(begin_time);
    /*
    cerr
      << filename << "\t"
      << rec->pos.size() << "games\t"
      << finish_time << "sec";
    for (auto& v : rec->pos)
      cerr << " " << v.size();
    cerr << endl;
    */
    return;
  }

  cerr << "load " << filename << endl;

  vector<char> buf;
  Inflate(filename, buf);
  boost::interprocess::basic_ivectorstream<vector<char>> in(buf);
  auto game = AllocateGame();
  InitializeBoard(game);
  float prob[362];
  int win;
  int color;

  int num = 0;
  int num_games = 0;
  int last_moves = 0;
  unsigned long long previous_hash = 0;
  while (!in.eof()) {
    size_t last_pos = in.tellg();
    win = 100;
    if (!ReadLzTrainingData(in, game, color, prob, win))
      break;
    /*
    cerr
      << "MOVES:" << game->moves
      << "\tHASH:" << game->previous1_hash
      << "\tHASH:" << game->current_hash
      << "\tCOLOR:" << color
      << "\tWIN:" << win << endl;
    */
    if (game->moves == 1 || game->previous1_hash != previous_hash) {
      /*
      cerr
        << "MOVES:" << game->moves
        << "\tLAST:" << last_moves
        << "\tCOLOR:" << color
        << "\tWIN:" << win << endl;
      */
      rec->pos.push_back({});
      num_games++;
    }
    rec->pos[rec->pos.size() - 1].push_back(last_pos);
    last_moves = game->moves;
    previous_hash = game->current_hash;
    ClearBoard(game);
    num++;
  }
  FreeGame(game);
  num_games++;
  auto finish_time = GetSpendTime(begin_time);
  cerr
    << filename << "\t"
    << num << "steps\t"
    << num_games << "games\t"
    << finish_time << "sec";
  for (auto& v : rec->pos)
    cerr << " " << v.size();
  cerr << endl;

  ofstream index_out(filename + ".idx");
  for (auto& v : rec->pos) {
    for (size_t i = 0; i < v.size(); i++) {
      if (i > 0)
        index_out << ' ';
      index_out << v[i];
    }
    index_out << endl;
  }
}

struct DataSet {
  size_t num_req;

  std::vector<float> basic;
  std::vector<float> features;
  std::vector<float> color;
  std::vector<float> win;
  std::vector<float> win2;
  std::vector<float> move;
  std::vector<float> statistic;
  std::vector<float> score;
  std::vector<float> score_value;
};

class Reader {
public:
  size_t current_rec;
  std::mt19937_64 mt;
  game_info_t* game;
  game_info_t* game_work;
  vector<LZ_record_t> *records;
  atomic<size_t> *records_cursor;

  explicit Reader(vector<LZ_record_t> *records, atomic<size_t> *records_cursor)
    : records(records), records_cursor(records_cursor) {
    random_device rd;
    mt.seed(rd());

    game = AllocateGame();
    InitializeBoard(game);
    game_work = AllocateGame();
    InitializeBoard(game_work);

    current_rec = 0;
  }

  ~Reader() {
    FreeGame(game);
    FreeGame(game_work);
  }

  bool Play(DataSet& data, const LZ_record_t& kifu, istream& in) {
    int num_game = mt() % kifu.pos.size();
    const vector<size_t>& file_pos = kifu.pos[num_game];
    int dump_turn = mt() % file_pos.size();
    if (use_easy_state && mt() % 2 == 0)
      dump_turn = min(file_pos.size() / 2 + mt() % file_pos.size() / 2, file_pos.size() - 1);
    in.seekg(file_pos[dump_turn]);

    float prob[362];
    int color;
    int win_color;
    ReadLzTrainingData(in, game, color, prob, win_color);

    int player_color = 0;
    //SetHandicapNum(0);
    //SetKomi(kifu.komi);
    //ClearBoard(game);

    //cerr << "RAND " << dump_turn << endl;
    //PrintBoard(game);

    /*
    cerr << "RAND " << FormatMove(pos) << endl;
    PrintBoard(game);
    AnalyzePoRating(game, color, rate);
    DumpFeature(&store_node, color, move, win_color == color ? 1 : -1, game, trans, true);
    //dump_turn += 3;
    trans++;
    */
    int my_color = color;
    int trans = mt() % 8;
    WritePlanes(data.basic, data.features,
      game, game_work, color, trans);
    data.color.push_back(color - 1);

    int ofs = data.move.size();
    data.move.resize(ofs + pure_board_max + 1);
    for (int i = 0; i < pure_board_max; i++) {
      int moveT = RevTransformMove(onboard_pos[i], trans);
      data.move[ofs + PureBoardPos(moveT)] = prob[i];
    }
    data.move[ofs + pure_board_max] = prob[pure_board_max];

    if (win_color == S_BLACK) {
      if (use_smooth_value)
        data.win.push_back(min(1.0f, 0.5f + (dump_turn + 1.0f) / file_pos.size()));
      else
        data.win.push_back(1.0f);
    } else if (win_color == S_WHITE) {
      if (use_smooth_value)
        data.win.push_back(-min(1.0f, 0.5f + (dump_turn + 1.0f) / file_pos.size()));
      else
        data.win.push_back(-1);
    } else
      data.win.push_back(0);

    in.seekg(file_pos[file_pos.size() - 1]);
    ReadLzTrainingData(in, game_work, color, prob, win_color);
    //PrintBoard(game);

    size_t stat_ofs = data.statistic.size();
    data.statistic.resize(stat_ofs + pure_board_max);

    for (int j = 0; j < pure_board_max; j++) {
      int pos = TransformMove(onboard_pos[j], trans);
      int c = game_work->board[pos];
      float o;
      if (c == S_EMPTY)
        o = 0;
      else if (c == S_BLACK)
        o = 1;
      else
        o = -1;
      data.statistic[stat_ofs + j] += o;
    }

    size_t score_ofs = data.score.size();
    data.score.resize(score_ofs + SCORE_DIM);
    int num_sim = 2;
    float sum = 0;

    for (int k = 0; k < num_sim; k++) {
      CopyGame(game, game_work);
      Simulation(game, color, &mt);

      float score = 0;
      //PrintBoard(game);
      for (int j = 0; j < pure_board_max; j++) {
        int pos = TransformMove(onboard_pos[j], trans);
        int c = game->board[pos];
        if (c == S_EMPTY)
          c = territory[Pat3(game->pat, pos)];
        float o;
        if (c == S_EMPTY)
          o = 0;
        else if (c == S_BLACK)
          o = 1;
        else
          o = -1;
        sum += o;
        score += o;
        //data.statistic[stat_ofs + j] += o;
      }


      //data.score.push_back(score);
      int score_label = round(score - SCORE_OFFSET);
      if (score_label < 0)
        score_label = 0;
      if (score_label >= SCORE_DIM)
        score_label = SCORE_DIM - 1;
      for (int j = 0; j < SCORE_DIM; j++) {
        data.score[score_ofs + j] += (score_label == j ? 1.0f : 0.0f) / num_sim;
      }
    }
#if 0
    for (int j = 0; j < pure_board_max; j++)
      data.statistic[stat_ofs + j] /= num_sim;
#endif
    double score = sum / num_sim;

#if 0
    if (score != sum) {
      cerr << "################################################################################" << endl;
      cerr << kifu.filename << endl;
      cerr << "sum:" << sum << " score:" << score << endl;
      PrintBoard(game);

      ClearBoard(game);
      int color = S_BLACK;
      for (int i = 0; i < kifu.moves - 1; i++) {
        int pos = GetKifuMove(&kifu, i);
        PutStone(game, pos, color);
        color = FLIP_COLOR(color);
      }
      PrintBoard(game);

      extern void Simulation(game_info_t *game, int starting_color, std::mt19937_64 *mt, bool print);
      Simulation(game, color, &mt, true);
      PrintBoard(game);
    }
#endif

    //cerr << "sum:" << sum << " score:" << score << endl;
    static std::atomic<int64_t> score_error;
    //static std::atomic<int64_t> sum_komi;
    static std::atomic<int64_t> score_num;
    static std::atomic<int64_t> win_num;
    static std::atomic<int64_t> win_error;
    std::atomic_fetch_add(&score_num, (int64_t)1);
    std::atomic_fetch_add(&score_error, (int64_t)abs(score - sum));
    if (game_work->moves > 250) {
      std::atomic_fetch_add(&win_num, (int64_t)1);
      if (score > komi[0]) {
        if (win_color == S_WHITE)
          std::atomic_fetch_add(&win_error, (int64_t)1);
      }
      else {
        if (win_color == S_BLACK)
          std::atomic_fetch_add(&win_error, (int64_t)1);
      }
    }
#if 0
    {
      lock_guard<mutex> lock(extract_mutex);
      PrintBoard(game_work);

      in.seekg(file_pos[file_pos.size() - 1]);
      ReadLzTrainingData(in, game_work, color, prob, win_color);
      PrintBoard(game_work);

      PrintBoard(game);
      cerr << "win:" << win_color << " score:" << (score - komi[0]) << endl;
    }
#endif
    //std::atomic_fetch_add(&sum_komi, (int64_t)kifu.komi);
    if (score_num % 100000 == 0) {
      cerr << "score error: " << (score_error / (double)score_num)
        //<< " komi:" << (sum_komi / (double)score_num)
        << " win:" << (win_error / (double)win_num)
        << endl;
    }

    for (int j = 0; j < SCORE_DIM; j++) {
      float komi = SCORE_WIN_OFFSET + j;
      if (score > komi)
        data.win2.push_back(1);
      else
        data.win2.push_back(-1);
    }

    data.score_value.push_back(score);

    //cerr << "RE:" << kifu.result << " color:" << my_color << " RAND:" << kifu.random_move << " sum:" << sum << endl;
    data.num_req++;
    return true;
  }

  void ReadN(DataSet& data, size_t num_rest, size_t per_file) {
    auto cursor = std::atomic_fetch_add(records_cursor, (size_t)1);
    LZ_record_t* rec = &(*records)[cursor % records->size()];
    if (false) {
      lock_guard<mutex> lock(extract_mutex);
      cerr << "load " << cursor << " " << rec->filename << endl;
    }

    if (rec->pos.size() == 0) {
      ExtractKifu(rec);
    }

    vector<char> buf;
    try {
      Inflate(rec->filename, buf);
    }
    catch (...) {
      cerr << "failed to load " << rec->filename << endl;
    }
    boost::interprocess::basic_ivectorstream<vector<char>> in(buf);

    size_t num_read = min(num_rest, per_file > 0 ? per_file : rec->pos.size());

    for (size_t n = 0; n < num_read; n++) {
      Play(data, *rec, in);
    }
    current_rec++;
  }

  unique_ptr<DataSet> Read(size_t n, size_t per_file) {
    auto data = make_unique<DataSet>();
    data->num_req = 0;
    data->basic.reserve(pure_board_max * 18 * n);
    data->features.reserve(pure_board_max * 90 * n);
    data->color.reserve(n);
    data->win.reserve(n);
    data->win2.reserve(SCORE_DIM * n);
    data->move.reserve(pure_board_max * n);
    data->statistic.reserve(pure_board_max * n);
    data->score.reserve(SCORE_DIM * n);
    data->score_value.reserve(n);

    while (data->num_req < n) {
      ReadN(*data, n - data->num_req, per_file);
      /*
      cerr << data->num_req
        << " basic:" << data->basic.size()
        << " win:" << data->win.size()
        << " score:" << data->score.size()
        << endl;
      */
    }

    return data;
  }
};


class MinibatchReader {
public:
  FunctionPtr net;
  CNTK::Variable var_basic, var_features;
  CNTK::Variable var_color, var_komi, var_statistic;
  CNTK::Variable var_win, var_move;
  CNTK::Variable var_score;
  CNTK::Variable var_score_value;
  CNTK::Variable var_win2;

  explicit MinibatchReader(FunctionPtr net)
    : net(net) {
    random_device rd;

    GetInputVariableByName(net, L"basic", var_basic);
    GetInputVariableByName(net, L"features", var_features);
    GetInputVariableByName(net, L"color", var_color);
    GetInputVariableByName(net, L"komi", var_komi);

    GetInputVariableByName(net, L"win", var_win);
    GetInputVariableByName(net, L"win2", var_win2);
    GetInputVariableByName(net, L"move", var_move);
    GetInputVariableByName(net, L"statistic", var_statistic);
    GetInputVariableByName(net, L"score", var_score);
    GetInputVariableByName(net, L"score_value", var_score_value);

    //CNTK::Variable var_ol;
    //GetOutputVaraiableByName(net, L"ol_L2", var_ol);
  }

  std::unordered_map<CNTK::Variable, CNTK::ValuePtr> GetMiniBatchData(DataSet& data) {
    size_t num_req = data.num_req;
    //CNTK::NDShape shape_komi = var_komi.Shape().AppendShape({ 1, num_req });
    //CNTK::ValuePtr value_komi = CNTK::MakeSharedObject<CNTK::Value>(CNTK::MakeSharedObject<CNTK::NDArrayView>(shape_komi, data.komi, true));

    CNTK::NDShape shape_move = var_move.Shape().AppendShape({ 1, num_req });
    CNTK::ValuePtr value_move = CNTK::MakeSharedObject<CNTK::Value>(CNTK::MakeSharedObject<CNTK::NDArrayView>(shape_move, data.move, true));
    if (shape_move.TotalSize() != data.move.size()) {
      cerr << "bad move size" << endl;
      abort();
    }

    std::unordered_map<CNTK::Variable, CNTK::ValuePtr> inputs = {
      { var_move, value_move },
    };

    if (var_basic.IsInitialized()) {
      CNTK::NDShape shape_basic = var_basic.Shape().AppendShape({ 1, num_req });
      CNTK::ValuePtr value_basic = CNTK::MakeSharedObject<CNTK::Value>(CNTK::MakeSharedObject<CNTK::NDArrayView>(shape_basic, data.basic, true));
      if (shape_basic.TotalSize() != data.basic.size()) {
        cerr << "bad basic size" << endl;
        abort();
      }
      inputs[var_basic] = value_basic;
    }
    if (var_features.IsInitialized()) {
      CNTK::NDShape shape_features = var_features.Shape().AppendShape({ 1, num_req });
      CNTK::ValuePtr value_features = CNTK::MakeSharedObject<CNTK::Value>(CNTK::MakeSharedObject<CNTK::NDArrayView>(shape_features, data.features, true));
      if (shape_features.TotalSize() != data.features.size()) {
        cerr << "bad features size" << endl;
        abort();
      }
      inputs[var_features] = value_features;
    }

    if (var_color.IsInitialized()) {
      CNTK::NDShape shape_color = var_color.Shape().AppendShape({ 1, num_req });
      CNTK::ValuePtr value_color = CNTK::MakeSharedObject<CNTK::Value>(CNTK::MakeSharedObject<CNTK::NDArrayView>(shape_color, data.color, true));
      if (shape_color.TotalSize() != data.color.size()) {
        cerr << "bad color size" << endl;
        abort();
      }
      inputs[var_color] = value_color;
    }

    if (var_win.IsInitialized()) {
      CNTK::NDShape shape_win = var_win.Shape().AppendShape({ 1, num_req });
      CNTK::ValuePtr value_win = CNTK::MakeSharedObject<CNTK::Value>(CNTK::MakeSharedObject<CNTK::NDArrayView>(shape_win, data.win, true));
      if (shape_win.TotalSize() != data.win.size()) {
        cerr << "bad win size" << endl;
        abort();
      }
      inputs[var_win] = value_win;
    }
    if (var_win2.IsInitialized()) {
      CNTK::NDShape shape_win2 = var_win2.Shape().AppendShape({ 1, num_req });
      CNTK::ValuePtr value_win2 = CNTK::MakeSharedObject<CNTK::Value>(CNTK::MakeSharedObject<CNTK::NDArrayView>(shape_win2, data.win2, true));
      if (shape_win2.TotalSize() != data.win2.size()) {
        cerr << "bad win2 size" << endl;
        abort();
      }
      inputs[var_win2] = value_win2;
    }
    if (var_statistic.IsInitialized()) {
      CNTK::NDShape shape_statistic = var_statistic.Shape().AppendShape({ 1, num_req });
      CNTK::ValuePtr value_statistic = CNTK::MakeSharedObject<CNTK::Value>(CNTK::MakeSharedObject<CNTK::NDArrayView>(shape_statistic, data.statistic, true));
      if (shape_statistic.TotalSize() != data.statistic.size()) {
        cerr << "bad statistics size" << endl;
        abort();
      }
      inputs[var_statistic] = value_statistic;
    }
    if (var_score.IsInitialized()) {
      CNTK::NDShape shape_score = var_score.Shape().AppendShape({ 1, num_req });
      CNTK::ValuePtr value_score = CNTK::MakeSharedObject<CNTK::Value>(CNTK::MakeSharedObject<CNTK::NDArrayView>(shape_score, data.score, true));
      if (shape_score.TotalSize() != data.score.size()) {
        cerr << "bad score size" << endl;
        abort();
      }
      inputs[var_score] = value_score;
    }
    if (var_score_value.IsInitialized()) {
      CNTK::NDShape shape_score_value = var_score_value.Shape().AppendShape({ 1, num_req });
      CNTK::ValuePtr value_score_value = CNTK::MakeSharedObject<CNTK::Value>(CNTK::MakeSharedObject<CNTK::NDArrayView>(shape_score_value, data.score_value, true));
      if (shape_score_value.TotalSize() != data.score_value.size()) {
        cerr << "bad score_value size" << endl;
        abort();
      }
      inputs[var_score_value] = value_score_value;
    }

    return inputs;
  }
};


static map<LZ_record_t*, shared_ptr<float[]>> statistic;
extern int threads;

static vector<string>
ListFiles( const std::string& kifu_dir )
{
  vector<string> filenames;
  stringstream ss{ kifu_dir };
  std::string dir;
  while (getline(ss, dir, ',')) {
    cerr << "Reading from " << dir << endl;
    size_t count = filenames.size();
    for (auto entry : fs::recursive_directory_iterator(dir)) {
      if (entry.path().extension() == ".gz") {
        filenames.push_back(entry.path().string());
        //cerr << "Read " << entry.path().string() << endl;
        //if (records.size() > 1000) break;
      }
    }
    cerr << ">> " << (filenames.size() - count) << endl;
  }

  cerr << "OK " << filenames.size() << endl;
  return filenames;
}

static volatile bool running;
static mutex mutex_queue;
static queue<unique_ptr<DataSet>> data_queue;

static mutex records_mutex;
static vector<LZ_record_t> records_train;
static atomic<size_t> records_cursor = {};

static int minibatch_size;

static void
ReadFiles( const vector<string>& filenames )
{
  cerr << "ReadFiles ";
  size_t size = filenames.size();
  vector<string> files;
  files.reserve(size);
  for (auto& s : filenames) {
    if (s.find("_low_") != std::string::npos
        && mt() % 10 > 0)
      continue;
    files.emplace_back(s);
  }
  shuffle(begin(files), end(files), mt);

  size = files.size();
  records_train.resize(size);
  for (size_t i = 0; i < size; i++) {
    auto& kifu = records_train[i];
    kifu.filename = files[i];
    kifu.pos.clear();
  }
  records_cursor = 0;
  cerr << records_train.size() << " from " << filenames.size() << endl;
}

void
ReadGames(int thread_no)
{
  Reader reader { &records_train, &records_cursor };
  while (running) {
    //cerr << "READ " << thread_no << endl;
    auto data = reader.Read(minibatch_size, 0);
    mutex_queue.lock();
    data_queue.push(move(data));

    while (data_queue.size() > 10 && running) {
      mutex_queue.unlock();
      this_thread::sleep_for(chrono::milliseconds(10));
      mutex_queue.lock();
    }
    mutex_queue.unlock();
  }
}

#define CHECK_FEATURE_MODE 0

static void
ReadCvData( vector<string>& filenames, vector<unique_ptr<DataSet>>& cv_data )
{
  vector<LZ_record_t> cv_records;
  vector<string> test_filenames;
  if (test_kifu_dir.size() > 0)
    test_filenames = ListFiles(test_kifu_dir);

  int cv_size;
  if (test_filenames.size() > 0) {
    cv_size = test_filenames.size() * num_games_per_file;
    cv_records.resize(test_filenames.size());
    for (auto i = 0; i < test_filenames.size(); i++) {
      cerr << "test " << test_filenames[i] << endl;
      cv_records[i].filename = test_filenames[i];
    }
  } else {
#if CHECK_FEATURE_MODE
    cv_size = 1024 * 100;
#else
    cv_size = 1024;
    // cv_size = 256;
#endif
    cv_records.resize(cv_size / num_games_per_file);
    for (int i = 0; i < cv_records.size(); i++) {
      if (i % 10000 == 0)
        cerr << "cv " << (100.0 * i / cv_size) << "%" << endl;
      auto f = filenames.back();
      filenames.pop_back();

      auto kifu = &cv_records[i];
      kifu->filename = f;
      cerr << "test " << f << endl;
    }
  }

  atomic<size_t> cv_cursor = {};
  Reader cv_reader{ &cv_records, &cv_cursor };
  cv_reader.mt.seed(0);
  for (int i = 0; i < (cv_size + minibatch_size - 1) / minibatch_size; i++) {
    int size = min(minibatch_size, (int)(cv_size - cv_data.size() * minibatch_size));
    cerr << "cv " << (100.0 * i / (cv_size / minibatch_size)) << "%"
      << " read " << size
      << endl;
    cv_data.push_back(move(cv_reader.Read(size, num_games_per_file)));
  }
}

void
Train()
{
  try {
#if 0
    {
      vector<char> buf;
      Inflate("C:\\tmp\\train_6ee288bd\\train_6ee288bd_1.gz", buf);
      boost::interprocess::basic_ivectorstream<vector<char>> in(buf);
      auto game = AllocateGame();
      InitializeBoard(game);
      float prob[362];
      int win;
      int color;

      auto begin_time = ray_clock::now();
      int num = 0;
      int num_games = 0;
      while (!in.eof()) {
        ReadLzTrainingData(in, game, color, prob, win);
        PrintBoard(game);
        cerr
          << "COLOR:" << color
          << "\tWIN:" << win << endl;
        if (game->moves == 1)
          num_games++;
        ClearBoard(game);
        num++;
      }
      auto finish_time = GetSpendTime(begin_time);
      cerr << num << "steps "
        << num_games << "games "
        << finish_time << endl;
    }
#endif
    random_device rd;
    std::mt19937_64 mt{ 0 };

    ReadWeights();
    auto device = GetDevice();
    auto net = GetPolicyNetwork();

    const size_t outputFrequencyInMinibatches = 50;
    const size_t trainingCheckpointFrequency = 500;
    const int stepsize = 100;
#if 0
    const double lr_min = 1.0e-6;
    const double lr_max = 2.0e-3;
#else
    const double lr_min = 1.0e-3;
    const double lr_max = 5.0e-2;
#endif
    //const double lr_max = 2.0e-5;

    const size_t loop_size = trainingCheckpointFrequency * 2;
    minibatch_size = 320;
    const size_t step = minibatch_size * loop_size / 2 / threads;


    vector<string> filenames = ListFiles(kifu_dir);
    shuffle(begin(filenames), end(filenames), mt);

    vector<unique_ptr<DataSet>> cv_data;
    ReadCvData(filenames, cv_data);

#if CHECK_FEATURE_MODE
    {
      std::vector<Parameter> parameters;
      for (auto p : net->Parameters()) {
        parameters.push_back(p);
      }

      LearningRateSchedule learningRatePerSample = TrainingParameterPerSampleSchedule(1e-10);
      AdditionalLearningOptions option;
      MinibatchReader reader{ net };

      Variable err_move, err_value2, err_score, err_owner, mse_score;
      GetVariableByName(net->Outputs(), L"err_move", err_move);
      GetVariableByName(net->Outputs(), L"err_value2", err_value2);
      GetVariableByName(net->Outputs(), L"err_score", err_score);
      GetVariableByName(net->Outputs(), L"err_owner", err_owner);
      GetVariableByName(net->Outputs(), L"mse_score", mse_score);
      Variable trainingLoss;
      GetVariableByName(net->Outputs(), L"ce", trainingLoss);

      DataSet data;
      for (int k = 0; k < 105; k++) {
        for (int side = 0; side < 2; side++) {
          cerr << "MASK " << k << endl;
          int mf = 0;
          for (int j = 0; j < 5; j++) {
            Variable err;
            if (j == 0)
              err = err_move;
            else if (j == 1)
              err = err_value2;
            else if (j == 2)
              err = err_score;
            /*
              else if (j == 3)
              err = err_owner;
              else if (j == 4)
              err = mse_score;
            */
            if (!err.IsInitialized())
              continue;
            auto tester = CreateTrainer(net, trainingLoss, err,
              { SGDLearner(parameters, learningRatePerSample, option) });

            double accumulatedError = 0;
            double error = 0;
            size_t totalNumberOfSamples = 0;
            size_t numberOfMinibatches = 0;
            uint64_t masked = 0;

            //auto checkpoint = m_cv.m_source->GetCheckpointState();
            for (int i = 0; i < cv_data.size(); i++) {
              const auto& ref = cv_data[i];
              data.num_req = ref->num_req;
#define COPY_VEC(a, b) {b.resize(a.size()); std::copy(a.begin(), a.end(), b.begin());}

              COPY_VEC(ref->basic, data.basic);
              COPY_VEC(ref->features, data.features);
              COPY_VEC(ref->history, data.history);
              COPY_VEC(ref->color, data.color);
              COPY_VEC(ref->komi, data.komi);
              COPY_VEC(ref->win, data.win);
              COPY_VEC(ref->win2, data.win2);
              COPY_VEC(ref->move, data.move);
              COPY_VEC(ref->statistic, data.statistic);
              COPY_VEC(ref->score, data.score);
              COPY_VEC(ref->score_value, data.score_value);

              int kk = k;
              kk -= 1;
              if (kk >= 0 && kk < 24) {
                mf = kk;
                if (data.basic.size() % (pure_board_max * 24) != 0) {
                  cerr << "********* BAD BASIC " << kk << " " << data.basic.size() << endl;
                }
                for (int i = 0; i < data.basic.size() / pure_board_max / 24; i++) {
                  for (int j = 0; j < pure_board_max; j++) {
                    int pos = j + (i * 24 + kk) * pure_board_max;
                    if (data.basic[pos] != side) {
                      masked++;
                    }
                    data.basic[pos] = side;
                  }
                }
              }
              kk -= 24;
              if (kk >= 0 && kk < 80) {
                mf = kk;
                if (data.features.size() % (pure_board_max * 80) != 0) {
                  cerr << "********* BAD FEATURES " << kk << " " << data.basic.size() << endl;
                }
                for (int i = 0; i < data.features.size() / pure_board_max / 80; i++) {
                  for (int j = 0; j < pure_board_max; j++) {
                    int pos = j + (i * 80 + kk) * pure_board_max;
                    if (data.features[pos] != side) {
                      masked++;
                    }
                    data.features[pos] = side;
                  }
                }
              }

              unordered_map<Variable, ValuePtr> minibatch = reader.GetMiniBatchData(data);
              error = tester->TestMinibatch(minibatch, device);
              accumulatedError += error * data.num_req;
              totalNumberOfSamples += data.num_req;
              numberOfMinibatches++;
            }
            fwprintf(stderr, L"CV %s\t%.8g"
                     "\t%d\t%d\t%d"
                     "\t%lld\n",
              err.AsString().c_str(),
              accumulatedError / totalNumberOfSamples,
              k, mf, side,
              masked);
          }
        }
      }
      abort();
    }
#endif

    //records.resize(step * threads);
    //records_b.resize(step * threads);
    ReadFiles(filenames);

    vector<unique_ptr<thread>> reader_handles;

    int start_step = 0;

    for (auto entry : fs::directory_iterator(".")) {
      if (entry.path().extension() != ".999")
        continue;
      auto name = entry.path().filename().string();
      if (name.find("feedForward.net") == 0) {
        int pos = name.find('.', 16);
        auto step = name.substr(16, pos - 16);
        int istep = stoi(step);
        cerr << name << " --> " << istep << endl;
        if (istep > start_step)
          start_step = istep;
      }
    }
    while (true) {
      const wstring ckpName = L"feedForward.net." + to_wstring(start_step + 1) + L"." + to_wstring(999);
      wcerr << "Try " << ckpName << endl;
      FILE* fp = _wfopen(ckpName.c_str(), L"r");
      if (fp) {
        fclose(fp);
        start_step++;
      } else {
        break;
      }
    }
    if (start_step > 1) {
      const wstring ckpName = L"feedForward.net." + to_wstring(start_step) + L"." + to_wstring(999);
      wcerr << "Restrat " << start_step << endl;
      net = CNTK::Function::Load(ckpName, device);
    } else if (fs::exists(L"./ref.bin")) {
      auto ref_net = CNTK::Function::Load(L"ref.bin", device);
      if (ref_net) {
        for (auto p : net->Parameters()) {
            wcerr << p.AsString() << " " << p.NeedsGradient();
          auto& name = p.AsString();
          if (
            true
            //name.find(L"model_move.") != wstring::npos
            //|| name.find(L"sqm.") != wstring::npos
            //name.find(L"owner.") != wstring::npos
            //&& (name.find(L"core.core2") == wstring::npos || name.find(L"x.x.x.x.x") == wstring::npos)
            //&& (name.find(L"core.") != wstring::npos
            //  || name.find(L"model.") != wstring::npos)
            //name.find(L"core.p2_L2.p2_L2.scale") != wstring::npos
            ) {
            wcerr << " LEARN";
            for (auto rp : ref_net->Parameters()) {
              if (name == rp.AsString()) {
                wcerr << " COPY ";
                p.SetValue(rp.GetValue());
                break;
              }
            }
          }
          wcerr << endl;
        }
      }
    }

    for (int alt = start_step;; alt++) {
      for (auto& t : reader_handles)
        t->join();
      reader_handles.clear();
      //auto finish_time = GetSpendTime(begin_time) * 1000;
      //cerr << "read sgf " << finish_time << endl;

      // vector<LZ_record_t> *records_next;
      // if (alt % 2 == 0) {
      //   records_current = &records_a;
      //   records_next = &records_b;
      // } else {
      //   records_current = &records_b;
      //   records_next = &records_a;
      // }

      // for (int i = 0; i < threads; i++) {
      //   reader_handles.push_back(make_unique<thread>(ReadFiles, i, (alt * threads + i) * step, step, records_next));
      // }

      if (alt == start_step) {
        continue;
      }

      {
        lock_guard<mutex> lock(records_mutex);
        if (records_cursor >= records_train.size()) {
          cerr << "cursor:" << (size_t)records_cursor
            << " size:" << records_train.size() << endl;
          ReadFiles(filenames);
        }
      }

      running = true;
      vector<unique_ptr<thread>> handles;
      for (int i = 0; i < threads; i++) {
        handles.push_back(make_unique<thread>(ReadGames, i));
      }

      Variable trainingLoss;
      //GetVariableByName(net->Outputs(), L"ce_2", trainingLoss);
      GetVariableByName(net->Outputs(), L"ce", trainingLoss);

      Variable err_move, err_value2, err_score, err_owner, mse_score;
      GetVariableByName(net->Outputs(), L"err_move", err_move);
      GetVariableByName(net->Outputs(), L"err_value2", err_value2);
      GetVariableByName(net->Outputs(), L"err_score", err_score);
      GetVariableByName(net->Outputs(), L"err_owner", err_owner);
      GetVariableByName(net->Outputs(), L"mse_score", mse_score);

      InitializeSearchSetting();
      InitializeUctHash();

      int total_count = 0;
      int learn_count = 0;
      std::vector<Parameter> parameters;
      for (auto p : net->Parameters()) {
        if (alt < start_step + 2)
          wcerr << p.AsString() << " " << p.NeedsGradient();
        auto& name = p.AsString();
        total_count++;
        if (
          true
          //name.find(L"model_move.") != wstring::npos
          //|| name.find(L"sqm.") != wstring::npos
          //name.find(L"owner.") != wstring::npos
          //&& (name.find(L"core.core2") == wstring::npos || name.find(L"x.x.x.x.x") == wstring::npos)
          //&& name.find(L"core.core2") == wstring::npos
          //name.find(L"core.p2_L2.p2_L2.scale") != wstring::npos
          //&& (name.find(L"core.x.x.") == wstring::npos
          //  && name.find(L"model.") == wstring::npos)
          ) {
          if (alt < start_step + 2)
            wcerr << " LEARN";
          parameters.push_back(p);
          learn_count++;
        }
        if (alt < start_step + 2)
          wcerr << endl;
      }

      MinibatchReader reader{ net };

      wcerr << "Learn " << learn_count << " of " << total_count << endl;

      //Variable classifierOutputVar;
      //FunctionPtr classifierOutput = classifierOutputVar;
      Variable prediction;
      switch (alt % 4) {
      case 0:
        GetOutputVaraiableByName(net, L"err_move", prediction);
        break;
      case 1:
        GetOutputVaraiableByName(net, L"err_value2", prediction);
        break;
      case 2:
        GetOutputVaraiableByName(net, L"err_owner", prediction);
        break;
      case 3:
        GetOutputVaraiableByName(net, L"err_score", prediction);
        break;
      }
      if (!prediction.IsInitialized())
        GetOutputVaraiableByName(net, L"ce", prediction);
      wcerr << prediction.AsString() << endl;

      //double rate = 2.0e-8 + 1.0e-3 / stepsize * (stepsize - abs(alt % (stepsize * 2) - stepsize));
      //double rate = 2.0e-7 + 8.0e-6 / stepsize * (stepsize - abs(alt % (stepsize * 2) - stepsize));
#if 0
      double lr_scale = pow(1.5, -alt / stepsize / 2.0);
      double lr_max0 = max(lr_max * lr_scale, lr_min);
      double rate = lr_min + (lr_max0 - lr_min) / stepsize * (stepsize - alt % stepsize);
      if (alt < stepsize * 2)
        rate = lr_min + (lr_max0 - lr_min) / stepsize * (stepsize - abs(alt % (stepsize * 2) - stepsize));
      rate *= lr_scale;
      use_easy_state = alt < stepsize * 4;
#else
      int max_step = 5;
      double llr_min = log(lr_min);
      double llr_max = log(lr_max);
      double lrate = llr_min + (llr_max - llr_min) / max_step * (max_step - alt / stepsize);
      double rate = exp(lrate);
      if (alt < stepsize)
        rate = lr_min + (lr_max - lr_min) / stepsize * (stepsize - abs(alt * 5 % (stepsize * 2) - stepsize));
      if (alt / stepsize > max_step) {
        double lr_scale = pow(1.5, -(alt - max_step * stepsize) / stepsize / 2.0);
        rate *= lr_scale;
      }
      cerr << "LR " << alt << " "
           << min(max_step, alt / stepsize) << " "
           << lrate << " "
           << rate << endl;
      use_easy_state = alt < stepsize;
      use_smooth_value = alt / stepsize <= max_step;
#endif
      cerr << rate << endl;
      cerr << use_easy_state << " " << use_smooth_value << endl;
      //LearningRateSchedule learningRate = TrainingParameterPerSampleSchedule(rate);
      LearningRateSchedule learningRate = LearningRateSchedule(rate);
      MomentumSchedule momentumSchedule = MomentumSchedule(0.9);
      //LearningRateSchedule learningRatePerSample = TrainingParameterPerSampleSchedule(4.00e-07);
      AdditionalLearningOptions option;
      option.l2RegularizationWeight = 0.0001;
      //auto minibatchSource = TextFormatMinibatchSource(L"SimpleDataTrain_cntk_text.txt", { { L"features", inputDim }, { L"labels", numOutputClasses } });
      auto trainer = CreateTrainer(net, trainingLoss, prediction, {
        MomentumSGDLearner(parameters, learningRate, momentumSchedule, DefaultUnitGainValue(), option)
      });

#if 0
      {
        const wstring ckpName = L"feedForward.net." + to_wstring(15499);
        trainer->RestoreFromCheckpoint(ckpName);
      }
#endif

      auto begin_time = ray_clock::now();
      double mb = 0;
      double trainLossValue = 0;
      double evaluationValue = 0;

      for (size_t i = 0; i < loop_size; ++i) {
        mutex_queue.lock();
        while (data_queue.empty()) {
          mutex_queue.unlock();
          this_thread::sleep_for(chrono::milliseconds(10));
          mutex_queue.lock();
        }
        auto data = move(data_queue.front());
        data_queue.pop();
        mutex_queue.unlock();
        //auto minibatchData = minibatchSource->GetNextMinibatch(minibatchSize, device);
        std::unordered_map<Variable, ValuePtr> arguments = reader.GetMiniBatchData(*data);
        std::unordered_map<Variable, ValuePtr> outputsToFetch;

        try {
          trainer->TrainMinibatch(arguments, false, outputsToFetch, device);
          PrintTrainingProgress(trainer, i, outputFrequencyInMinibatches);
        } catch (const std::exception& err) {
          fprintf(stderr, "EXCEPTION occurred: %s\n", err.what());
          abort();
        }

        mb += 1;
        trainLossValue += trainer->PreviousMinibatchLossAverage();
        evaluationValue += trainer->PreviousMinibatchEvaluationAverage();

        if ((i % trainingCheckpointFrequency) == (trainingCheckpointFrequency - 1))
        {
          double finish_time = GetSpendTime(begin_time);
          wcerr << finish_time << " " << finish_time / (minibatch_size * trainingCheckpointFrequency) << " per sample" << endl;
          fprintf(stderr, "CrossEntropy loss = %.8g, Evaluation criterion = %.8g\n", trainLossValue / mb, evaluationValue / mb);
          const wstring ckpName = L"feedForward.net." + to_wstring(alt) + L"." + to_wstring(i);
          trainer->SaveCheckpoint(ckpName);

          for (int j = 0; j < 5; j++) {
            Variable err;
            if (j == 0)
              err = err_move;
            else if (j == 1)
              err = err_value2;
            else if (j == 2)
              err = err_score;
            else if (j == 3)
              err = err_owner;
            else if (j == 4)
              err = mse_score;
            if (!err.IsInitialized())
              continue;
            auto tester = CreateTrainer(net, trainingLoss, err,
              { SGDLearner(parameters, learningRate, option) });

            // Cross validation
            double accumulatedError = 0;
            double error = 0;
            size_t totalNumberOfSamples = 0;
            size_t numberOfMinibatches = 0;

            //auto checkpoint = m_cv.m_source->GetCheckpointState();
            for (int i = 0; i < cv_data.size(); i++) {
              unordered_map<Variable, ValuePtr> minibatch = reader.GetMiniBatchData(*cv_data[i]);
              error = tester->TestMinibatch(minibatch, device);
              accumulatedError += error * cv_data[i]->num_req;
              totalNumberOfSamples += cv_data[i]->num_req;
              numberOfMinibatches++;
            }
            fwprintf(stderr, L"CV %s %.8g\n",
              err.AsString().c_str(),
              accumulatedError / totalNumberOfSamples);
          }

          //m_cv.m_source->RestoreFromCheckpoint(checkpoint);
          trainer->SummarizeTestProgress();

          // Resume
          trainer->RestoreFromCheckpoint(ckpName);
          mb = 0;
          trainLossValue = 0;
          evaluationValue = 0;

          begin_time = ray_clock::now();
        }
      }
      running = false;
      for (auto& t : handles)
        t->join();
    }

  } catch (const std::exception& err) {
    fprintf(stderr, "EXCEPTION occurred: %s\n", err.what());
    abort();
  }
}
