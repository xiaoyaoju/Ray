#ifndef _MOVECACHE_H_
#define _MOVECACHE_H_

#include <string>
#include <random>
#include <memory>
#include <mutex>

#include "GoBoard.h"


class LGRContext {
public:
  void store(const game_info_t* game, int pos);
private:
  uint32_t hash[MAX_MOVES];
  uint16_t lib[MAX_MOVES];
  uint32_t hash_last[MAX_MOVES];
  uint16_t lib_last[MAX_MOVES];

  friend class LGR;
};

class LGR {
public:
  LGR();
  void reset();
  int getLGRF1(int col, int pos1, const game_info_t* game);
  void setLGRF1(int col, int pos1, int val, uint32_t hash2);
  void clearLGRF1(int col, int pos1);

  void setTGR1(int col, int pos1, int pos, uint32_t hash_last, uint32_t hash, uint16_t lib_last, uint16_t lib);
  int getTGR1(int col, int pos1, const game_info_t* game);

  int getLGRF2(int col, int pos1, int pos2);
  void setLGRF2(int col, int pos1, int pos2, int pos);
  bool hasLGRF2(int col, int pos1, int pos2);
  void clearLGRF2(int col, int pos1, int pos2);

  void update(game_info_t* game, int strat, int win, const LGRContext& ctx);

private:
  std::mutex board_mutex[BOARD_MAX];
  std::unique_ptr<int16_t[]> tgr1;
  std::unique_ptr<uint32_t[]> tgr1_hash;
  std::unique_ptr<uint16_t[]> tgr1_count;
  std::unique_ptr<uint32_t[]> tgr1_last;
  std::unique_ptr<uint16_t[]> tgr1_lib;
  std::unique_ptr<uint16_t[]> tgr1_lib_last;

  std::unique_ptr<uint32_t[]> lgrf1;
  std::unique_ptr<uint32_t[]> lgrf1_last;
  std::unique_ptr<int[]> lgrf2;
};

extern int tgr1_rate;
extern int lgrf1_rate;
extern bool use_lgrf2;
const uint16_t tgr_num = 8;

#endif
