#include "Train.h"
#include "UctSearch.h"
#include "SGFExtractor.hpp"
#include "DynamicKomi.h"
#include "Rating.h"
#include "Message.h"
#include "MoveCache.h"
#include "Simulation.h"
#include "Utility.h"

#include <filesystem>
#include <iostream>
#include <mutex>
#include <queue>
#include <random>
#include <thread>
#include <vector>

using namespace std;
using namespace CNTK;

#ifdef  _WIN32
namespace fs = std::experimental::filesystem;
#endif

static string kifu_dir = "kifu";

void
SetKifuDirectory(const std::string dir)
{
  kifu_dir = dir;
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

struct DataSet {
  size_t num_req;

  std::vector<float> basic;
  std::vector<float> features;
  std::vector<float> history;
  std::vector<float> color;
  std::vector<float> komi;
  std::vector<float> win;
  std::vector<float> move;
  std::vector<float> statistic;
};

static vector<string> filenames;
static mutex mutex_records;
//static vector<SGF_record> *records_next;
static vector<SGF_record> *records_current;
static vector<SGF_record> records_a;
static vector<SGF_record> records_b;
static map<SGF_record*, shared_ptr<float[]>> statistic;
extern int threads;

static void
ListFiles()
{
  for (auto entry : fs::recursive_directory_iterator(kifu_dir)) {
    if (entry.path().extension() == ".sgf") {
      filenames.push_back(entry.path().string());
      //cerr << "Read " << entry.path().string() << endl;
      //if (records.size() > 1000) break;
    }
  }

  cerr << "OK " << filenames.size() << endl;
}

static void
ReadFiles(int thread_no, size_t offset, size_t size, vector<SGF_record> *records)
{
  vector<string> files;
  files.reserve(size);
  for (size_t i = 0; i < size; i++) {
    files.push_back(filenames[(i + offset) % filenames.size()]);
  }
  random_device rd;
  std::mt19937_64 mt{ rd() };
  shuffle(begin(files), end(files), mt);

  for (size_t i = 0; i < size; i++) {
    auto kifu = &(*records)[i * threads + thread_no];
    ExtractKifu(files[i].c_str(), kifu);
    if (kifu->moves == 0)
      cerr << "Bad file " << files[i] << endl;

    if (kifu->moves > PURE_BOARD_MAX * 0.9)
      kifu->moves = PURE_BOARD_MAX * 0.9;
  }
}

class Reader {
public:
  size_t current_rec;
  std::mt19937_64 mt;
  game_info_t* game;
  int offset;

  explicit Reader(int offset)
    : offset(offset) {
    random_device rd;
    mt.seed(rd());

    game = AllocateGame();
    InitializeBoard(game);

    current_rec = 0;
  }

  ~Reader() {
    FreeGame(game);
  }

  bool Play(DataSet& data, const SGF_record& kifu) {
    int win_color;
    if (kifu.handicaps > 0)
      return false;
    switch (kifu.result) {
    case R_BLACK:
      win_color = S_BLACK;
      break;
    case R_WHITE:
      win_color = S_WHITE;
      break;
    default:
      return false;
    }

    int player_color = 0;
    SetHandicapNum(0);
    SetKomi(kifu.komi);
    ClearBoard(game);

    stringstream out;

    // Replay to random turn
    int dump_turn;
    if (kifu.random_move < 0) {
      uniform_int_distribution<int> dist_turn(1, max(1, kifu.moves - 20));
      dump_turn = dist_turn(mt);
    } else {
      //dump_turn = kifu.random_move - 1;
      uniform_int_distribution<int> dist_turn(kifu.random_move - 1, min(kifu.random_move + 8, kifu.moves - 1));
      dump_turn = dist_turn(mt);
    }

    int color = S_BLACK;
    double rate[PURE_BOARD_MAX];
    LGR lgr;
    LGRContext ctx;
    for (int i = 0; i < kifu.moves - 1; i++) {
      int pos = GetKifuMove(&kifu, i);
      //PrintBoard(game);
      //cerr << "#" << i << " " << FormatMove(pos) << " " << color << endl;
      if (!IsLegal(game, pos, color))
        return false;
      PutStone(game, pos, color);
      color = FLIP_COLOR(color);
      //PrintBoard(game);

      int move = GetKifuMove(&kifu, i + 1);

      if (move != PASS && move != RESIGN) {
        int ts[] = { 0, 1, 2, 3, 4, 5, 6, 7 };
        shuffle(begin(ts), end(ts), mt);
        AnalyzePoRating(game, color, rate);
#if 0
        DumpFeature(&store_node, color, move, win_color == color ? 1 : -1, game, ts[0], true);
#elif 0
        if (i < 10) {
          for (int t = 0; t < 4; t++) {
            DumpFeature(nullptr, color, move, win_color == color ? 1 : -1, game, ts[t], false);
          }
        } else if (i < 50) {
          for (int t = 0; t < 2; t++) {
            DumpFeature(nullptr, color, move, win_color == color ? 1 : -1, game, ts[t], false);
          }
        } else {
          DumpFeature(nullptr, color, move, win_color == color ? 1 : -1, game, ts[0], false);
        }
#else
        if (dump_turn == i) {
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
          WritePlanes(data.basic, data.features, data.history,
            nullptr, game, nullptr, color, trans);
          data.color.push_back(color - 1);
          data.komi.push_back(kifu.komi);
          int moveT = RevTransformMove(move, trans);

#if 0
          int ofs = data.move.size();
          data.move.resize(ofs + pure_board_max);
          if (kifu.comment[i + 1].size() > 0) {
            stringstream in{ kifu.comment[i + 1] };
            vector<float> rate;
            rate.reserve(PURE_BOARD_MAX);
            while (!in.eof()) {
              float r;
              in >> r;
              if (in.fail() || in.eof())
                break;
              in.ignore(1, ' ');
              rate.push_back(r);
            }
            for (int j = 0; j < pure_board_max; j++) {
              int pos = RevTransformMove(onboard_pos[j], trans);
              data.move[ofs + PureBoardPos(pos)] = rate[j];
            }
            //cerr << "Read rate " << data.move.size() << endl;
            if (rate.size() != pure_board_max) {
              cerr << "bad rate" << endl;
              cerr << kifu.comment[i + 1] << endl;
              for (int i = 0; i < data.move.size(); i++)
                cerr << data.move[i] << " ";
              cerr << endl;
              break;
            }
          } else {
            data.move[ofs + PureBoardPos(moveT)] = 1;
          }
#else
          int x = CORRECT_X(moveT) - 1;
          int y = CORRECT_Y(moveT) - 1;
          int label = x + y * pure_board_size;
          for (int j = 0; j < pure_board_max; j++) {
#if 1
            data.move.push_back(label == j ? 1.0f : 0.0f);
#else
            if (color == win_color)
              data.move.push_back(label == j ? 1.0f : 0.0f);
            else
              data.move.push_back(0.0f);
#endif
          }
#endif
          data.win.push_back(win_color == color ? 1 : -1);
          i++;
          for (; i < kifu.moves - 1; i++) {
            int pos = GetKifuMove(&kifu, i);
            if (!IsLegal(game, pos, color))
              return false;
            PutStone(game, pos, color);
            color = FLIP_COLOR(color);
          }
          Simulation(game, color, &mt);

          //PrintBoard(game);
          float sum = 0;
          for (int j = 0; j < pure_board_max; j++) {
            int pos = TransformMove(onboard_pos[j], trans);
            int c = game->board[pos];
            if (c == S_EMPTY)
              c = territory[Pat3(game->pat, pos)];
            float o;
            if (c == S_EMPTY)
              o = 0;
            else if (c == my_color)
              o = 1;
            else
              o = -1;
            sum += o;
            data.statistic.push_back(o);
          }
          //cerr << "RE:" << kifu.result << " color:" << my_color << " RAND:" << kifu.random_move << " sum:" << sum << endl;
          data.num_req++;
          return true;
        }
#endif
      }
    }
    return false;
  }

  void ReadOne(DataSet& data) {
    auto& r = (*records_current)[(current_rec * threads + offset) % records_current->size()];
    Play(data, r);

    current_rec++;
  }

  unique_ptr<DataSet> Read(size_t n) {
    auto data = make_unique<DataSet>();
    data->num_req = 0;
    data->basic.reserve(pure_board_max * 10 * n);
    data->features.reserve(pure_board_max * (F_MAX1 + F_MAX2) * n);
    data->history.reserve(pure_board_max * n);
    data->color.reserve(n);
    data->komi.reserve(n);
    data->win.reserve(n);
    data->move.reserve(pure_board_max * n);
    data->statistic.reserve(pure_board_max * n);

    while (data->num_req < n) {
      ReadOne(*data);
    }

    return data;
  }
};


class MinibatchReader {
public:
  FunctionPtr net;
  CNTK::Variable var_basic, var_features, var_history, var_color, var_komi, var_statistic;
  CNTK::Variable var_win, var_move;

  explicit MinibatchReader(FunctionPtr net)
    : net(net) {
    random_device rd;

    GetInputVariableByName(net, L"basic", var_basic);
    GetInputVariableByName(net, L"features", var_features);
    GetInputVariableByName(net, L"history", var_history);
    GetInputVariableByName(net, L"color", var_color);
    GetInputVariableByName(net, L"komi", var_komi);

    GetInputVariableByName(net, L"win", var_win);
    GetInputVariableByName(net, L"move", var_move);
    GetInputVariableByName(net, L"statistic", var_statistic);

    //CNTK::Variable var_ol;
    //GetOutputVaraiableByName(net, L"ol_L2", var_ol);
  }

  std::unordered_map<CNTK::Variable, CNTK::ValuePtr> GetMiniBatchData(DataSet& data) {
    size_t num_req = data.num_req;
    CNTK::NDShape shape_basic = var_basic.Shape().AppendShape({ 1, num_req });
    CNTK::ValuePtr value_basic = CNTK::MakeSharedObject<CNTK::Value>(CNTK::MakeSharedObject<CNTK::NDArrayView>(shape_basic, data.basic, true));
    CNTK::NDShape shape_features = var_features.Shape().AppendShape({ 1, num_req });
    CNTK::ValuePtr value_features = CNTK::MakeSharedObject<CNTK::Value>(CNTK::MakeSharedObject<CNTK::NDArrayView>(shape_features, data.features, true));
    CNTK::NDShape shape_history = var_history.Shape().AppendShape({ 1, num_req });
    CNTK::ValuePtr value_history = CNTK::MakeSharedObject<CNTK::Value>(CNTK::MakeSharedObject<CNTK::NDArrayView>(shape_history, data.history, true));
    CNTK::NDShape shape_color = var_color.Shape().AppendShape({ 1, num_req });
    CNTK::ValuePtr value_color = CNTK::MakeSharedObject<CNTK::Value>(CNTK::MakeSharedObject<CNTK::NDArrayView>(shape_color, data.color, true));
    CNTK::NDShape shape_komi = var_komi.Shape().AppendShape({ 1, num_req });
    CNTK::ValuePtr value_komi = CNTK::MakeSharedObject<CNTK::Value>(CNTK::MakeSharedObject<CNTK::NDArrayView>(shape_komi, data.komi, true));

    CNTK::NDShape shape_win = var_win.Shape().AppendShape({ 1, num_req });
    CNTK::ValuePtr value_win = CNTK::MakeSharedObject<CNTK::Value>(CNTK::MakeSharedObject<CNTK::NDArrayView>(shape_win, data.win, true));
    CNTK::NDShape shape_move = var_move.Shape().AppendShape({ 1, num_req });
    CNTK::ValuePtr value_move = CNTK::MakeSharedObject<CNTK::Value>(CNTK::MakeSharedObject<CNTK::NDArrayView>(shape_move, data.move, true));
    CNTK::NDShape shape_statistic = var_statistic.Shape().AppendShape({ 1, num_req });
    CNTK::ValuePtr value_statistic = CNTK::MakeSharedObject<CNTK::Value>(CNTK::MakeSharedObject<CNTK::NDArrayView>(shape_statistic, data.statistic, true));

    std::unordered_map<CNTK::Variable, CNTK::ValuePtr> inputs = {
      { var_basic, value_basic },
      { var_features, value_features },
      { var_history, value_history },
      { var_color, value_color },
      { var_komi, value_komi },
      { var_win, value_win },
      { var_move, value_move },
      { var_statistic , value_statistic },
    };

    return inputs;
  }
};

static volatile bool running;
static mutex mutex_queue;
static queue<unique_ptr<DataSet>> data_queue;

static int minibatch_size;

void
ReadGames(int thread_no)
{
  Reader reader{ thread_no };
  while (running) {
    //cerr << "READ " << thread_no << endl;
    auto data = reader.Read(minibatch_size);
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

void
Train()
{
  try {
    random_device rd;
    std::mt19937_64 mt{ rd() };

    ReadWeights();
    auto device = GetDevice();
    auto net = GetPolicyNetwork();

    ListFiles();
    shuffle(begin(filenames), end(filenames), mt);

    const size_t outputFrequencyInMinibatches = 50;
    const size_t trainingCheckpointFrequency = 500;
    const int stepsize = 200;
    const double lr_min = 1.0e-6;
    //const double lr_max = 2.0e-4;
    const double lr_max = 2.0e-4;

    const size_t loop_size = trainingCheckpointFrequency * 2;
    minibatch_size = 128;
    const size_t step = minibatch_size * loop_size / 2 / threads;

    records_a.resize(step * threads);
    records_b.resize(step * threads);

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
    }

    for (int alt = start_step;; alt++) {
      for (auto& t : reader_handles)
        t->join();
      reader_handles.clear();
      //auto finish_time = GetSpendTime(begin_time) * 1000;
      //cerr << "read sgf " << finish_time << endl;

      vector<SGF_record> *records_next;
      if (alt % 2 == 0) {
        records_current = &records_a;
        records_next = &records_b;
      } else {
        records_current = &records_b;
        records_next = &records_a;
      }

      for (int i = 0; i < threads; i++) {
        reader_handles.push_back(make_unique<thread>(ReadFiles, i, (alt * threads + i) * step, step, records_next));
      }

      if (alt == start_step) {
        continue;
      }

      running = true;
      vector<unique_ptr<thread>> handles;
      for (int i = 0; i < threads; i++) {
        handles.push_back(make_unique<thread>(ReadGames, i));
      }

      Variable trainingLoss;
      //GetVariableByName(net->Outputs(), L"ce_2", trainingLoss);
      GetVariableByName(net->Outputs(), L"ce", trainingLoss);

      Variable err_move, err_value2;
      GetVariableByName(net->Outputs(), L"errs_move", err_move);
      GetVariableByName(net->Outputs(), L"err_value2", err_value2);

      InitializeSearchSetting();
      InitializeUctHash();

      int total_count = 0;
      int learn_count = 0;
      std::vector<Parameter> parameters;
      for (auto p : net->Parameters()) {
        if (alt < 2)
          wcerr << p.AsString() << " " << p.NeedsGradient();
        auto& name = p.AsString();
        total_count++;
        if (
          //true
          //name.find(L"model_move.") != wstring::npos
          //|| name.find(L"sqm.") != wstring::npos
          name.find(L"owner.") != wstring::npos
          //&& (name.find(L"core.core2") == wstring::npos || name.find(L"x.x.x.x.x") == wstring::npos)
          //&& name.find(L"core.core2") == wstring::npos
          //name.find(L"core.p2_L2.p2_L2.scale") != wstring::npos
          ) {
          if (alt < 2)
            wcerr << " LEARN";
          parameters.push_back(p);
          learn_count++;
        }
        if (alt < 2)
          wcerr << endl;
      }

      MinibatchReader reader{ net };

      wcerr << "Learn " << learn_count << " of " << total_count << endl;

      //Variable classifierOutputVar;
      //FunctionPtr classifierOutput = classifierOutputVar;
      Variable prediction;
      switch (alt % 3) {
      case 0:
        GetOutputVaraiableByName(net, L"errs_move", prediction);
        break;
      case 1:
        GetOutputVaraiableByName(net, L"err_value2", prediction);
        break;
      case 2:
        GetOutputVaraiableByName(net, L"err_owner", prediction);
        break;
      }
      wcerr << prediction.AsString() << endl;

      //double rate = 2.0e-8 + 1.0e-3 / stepsize * (stepsize - abs(alt % (stepsize * 2) - stepsize));
      //double rate = 2.0e-7 + 8.0e-6 / stepsize * (stepsize - abs(alt % (stepsize * 2) - stepsize));
      double lr_scale = pow(2, -alt / stepsize / 2.0);
      double rate = lr_min + (lr_max - lr_min) / stepsize * (stepsize - alt % stepsize);
      rate *= lr_scale;
      cerr << rate << endl;
      LearningRateSchedule learningRatePerSample = TrainingParameterPerSampleSchedule(rate);
      //LearningRateSchedule learningRatePerSample = TrainingParameterPerSampleSchedule(4.00e-07);
      AdditionalLearningOptions option;
      option.l2RegularizationWeight = 0.0001;
      //auto minibatchSource = TextFormatMinibatchSource(L"SimpleDataTrain_cntk_text.txt", { { L"features", inputDim }, { L"labels", numOutputClasses } });
      auto trainer = CreateTrainer(net, trainingLoss, prediction, { SGDLearner(parameters, learningRatePerSample, option) });

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
        trainer->TrainMinibatch(arguments, false, outputsToFetch, device);
        PrintTrainingProgress(trainer, i, outputFrequencyInMinibatches);

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
