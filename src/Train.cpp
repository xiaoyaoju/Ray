#include "Train.h"
#include "UctSearch.h"
#include "SGFExtractor.hpp"
#include "DynamicKomi.h"
#include "Rating.h"
#include "Message.h"
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
};

static vector<shared_ptr<SGF_record>> records;
extern int threads;

static void
ReadFile() {
  for (auto entry : fs::recursive_directory_iterator(kifu_dir)) {
    if (fs::is_regular_file(entry) && entry.path().extension() == ".sgf") {
      //cerr << "Read " << entry.path().string() << endl;

      auto kifu = make_shared<SGF_record>();
      ExtractKifu(entry.path().string().c_str(), kifu.get());

      records.push_back(kifu);
    }
  }

  cerr << "OK " << records.size() << endl;
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
      uniform_int_distribution<int> dist_turn(10, kifu.moves - 1);
      dump_turn = dist_turn(mt);
    } else {
      //dump_turn = kifu.random_move - 1;
      uniform_int_distribution<int> dist_turn(kifu.random_move - 1, min(kifu.random_move + 8, kifu.moves - 1));
      dump_turn = dist_turn(mt);
    }

    int color = S_BLACK;
    double rate[PURE_BOARD_MAX];
    for (int i = 0; i < kifu.moves - 1; i++) {
      int pos = GetKifuMove(&kifu, i);
      //PrintBoard(game);
      //cerr << "#" << i << " " << FormatMove(pos) << " " << color << endl;
      PutStone(game, pos, color);
      color = FLIP_COLOR(color);
      //PrintBoard(game);

      int move = GetKifuMove(&kifu, i + 1);

      if (move != PASS && move != RESIGN) {
        int ts[] = { 0, 1, 2, 3, 4, 5, 6, 7 };
        shuffle(begin(ts), end(ts), mt);
#if 0
        AnalyzePoRating(game, color, rate);
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
          int trans = mt() % 8;
          WritePlanes(data.basic, data.features, data.history,
            nullptr, game, nullptr, color, trans);
          data.color.push_back(color - 1);
          data.komi.push_back(kifu.komi);
          int moveT = RevTransformMove(move, trans);

#if 1
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
          data.num_req++;
          return true;
        }
#endif
      }
    }
    return false;
  }

  void ReadOne(DataSet& data) {
    auto r = records[(current_rec * threads + offset) % records.size()];
    Play(data, *r);

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

    while (data->num_req < n) {
      ReadOne(*data);
    }

    return data;
  }
};


class MinibatchReader {
public:
  FunctionPtr net;
  CNTK::Variable var_basic, var_features, var_history, var_color, var_komi;
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

    std::unordered_map<CNTK::Variable, CNTK::ValuePtr> inputs = {
      { var_basic, value_basic },
      { var_features, value_features },
      { var_history, value_history },
      { var_color, value_color },
      { var_komi, value_komi },
      { var_win, value_win },
      { var_move, value_move },
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

    ReadFile();

    minibatch_size = 128;

    for (int alt = 0;; alt++) {
      shuffle(begin(records), end(records), mt);

      running = true;
      vector<unique_ptr<thread>> handles;
      for (int i = 0; i < threads; i++) {
        handles.push_back(make_unique<thread>(ReadGames, i));
      }

      Variable trainingLoss;
      //GetVariableByName(net->Outputs(), L"ce_2_L2", trainingLoss);
      GetVariableByName(net->Outputs(), L"ce", trainingLoss);

      Variable errs_move_L2, err_value2_L2;
      GetVariableByName(net->Outputs(), L"errs_move_L2", errs_move_L2);
      GetVariableByName(net->Outputs(), L"err_value2_L2", err_value2_L2);

      InitializeSearchSetting();
      InitializeUctHash();

      int total_count = 0;
      int learn_count = 0;
      std::vector<Parameter> parameters;
      for (auto p : net->Parameters()) {
        if (alt == 0)
          wcerr << p.AsString() << " " << p.NeedsGradient();
        auto& name = p.AsString();
        total_count++;
        if (
          //true
          name.find(L"core.core0") == wstring::npos
          && name.find(L"core.core1") == wstring::npos
          //&& (name.find(L"core.core2") == wstring::npos || name.find(L"x.x.x.x.x") == wstring::npos)
          //&& name.find(L"core.core2") == wstring::npos
          //name.find(L"core.p2_L2.p2_L2.scale") != wstring::npos
          ) {
          if (alt == 0)
            wcerr << " LEARN";
          parameters.push_back(p);
          learn_count++;
        }
        if (alt == 0)
          wcerr << endl;
      }

      MinibatchReader reader{ net };

      wcerr << "Learn " << learn_count << " of " << total_count << endl;

      //Variable classifierOutputVar;
      //FunctionPtr classifierOutput = classifierOutputVar;
      Variable prediction;
      if (alt % 2 == 0)
        GetOutputVaraiableByName(net, L"errs_move_L2", prediction);
      else
        GetOutputVaraiableByName(net, L"err_value2_L2", prediction);
      wcerr << prediction.AsString() << endl;

      int stepsize = 100;
      double rate = 2.0e-7 + 8.0e-6 / stepsize * (stepsize - abs(alt % (stepsize * 2) - stepsize));
      //double rate = 2.0e-7 + 8.0e-6 / 100 * abs(100 - alt % 100);
      cerr << rate << endl;
      LearningRateSchedule learningRatePerSample = TrainingParameterPerSampleSchedule(rate);
      //LearningRateSchedule learningRatePerSample = TrainingParameterPerSampleSchedule(4.00e-07);
      AdditionalLearningOptions option;
      option.l2RegularizationWeight = 0.0001;
      //auto minibatchSource = TextFormatMinibatchSource(L"SimpleDataTrain_cntk_text.txt", { { L"features", inputDim }, { L"labels", numOutputClasses } });
      auto trainer = CreateTrainer(net, trainingLoss, prediction, { SGDLearner(parameters, learningRatePerSample, option) });
      size_t outputFrequencyInMinibatches = 50;
      size_t trainingCheckpointFrequency = 500;

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

      for (size_t i = 0; i < trainingCheckpointFrequency * 2; ++i) {
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
