#include <iostream>
#include <algorithm>
#include <functional>
#include <string>
#include <numeric>

#include "MoveCache.h"


using namespace std;

////////////////
//    変数    //
////////////////

// LGR
int tgr1_rate = 0;
int lgrf1_rate = 0;
bool use_lgrf2 = false;

static uint16_t GetLib(const game_info_t* game, int pos)
{
  uint16_t lib = 0;
  int neighbor[] = {
    NORTH(NORTH(pos)),
    NORTH_WEST(pos),
    WEST(WEST(pos)),
    SOUTH_WEST(pos),
    SOUTH(SOUTH(pos)),
    SOUTH_EAST(pos),
    EAST(EAST(pos)),
    NORTH_EAST(pos),
  };
  for (int i = 0; i < 8; i++) {
    int p = neighbor[i];
    auto col = game->board[p];
    if (col == S_BLACK || col == S_WHITE) {
      auto& s = game->string[game->string_id[p]];
      int l = min(s.libs, 4) - 1;
      lib |= (l << (i * 2));
    }
  }
  return lib;
}

// Based on Oakfoam http://oakfoam.com/

LGR::LGR()
{
  reset();
}

void
LGR::reset()
{
  tgr1.reset();
  tgr1_hash.reset();
  tgr1_count.reset();
  if (tgr1_rate > 0) {
    size_t s = 2 * pure_board_max;
    tgr1.reset(new int16_t[s * tgr_num]);
    fill_n(tgr1.get(), s * tgr_num, PASS);
    tgr1_hash.reset(new uint32_t[s * tgr_num]);
    tgr1_last.reset(new uint32_t[s * tgr_num]);
    tgr1_lib.reset(new uint16_t[s * tgr_num]);
    tgr1_lib_last.reset(new uint16_t[s * tgr_num]);
    tgr1_count.reset(new uint16_t[s]);
    fill_n(tgr1_count.get(), s, 0);
  }

  lgrf1.reset();
  if (lgrf1_rate > 0) {
    lgrf1.reset(new uint32_t[2 * pure_board_max]);
    fill_n(lgrf1.get(), 2 * pure_board_max, PASS);
    lgrf1_last.reset(new uint32_t[2 * pure_board_max]);
  }

  lgrf2.reset();
  if (use_lgrf2) {
    lgrf2.reset(new int[2 * pure_board_max * pure_board_max]);
    fill_n(lgrf2.get(), 2 * pure_board_max * pure_board_max, PASS);
  }
}

void
LGR::setTGR1(int col, int pos1, int val, uint32_t hash_last, uint32_t hash, uint16_t lib_last, uint16_t lib)
{
  lock_guard<mutex> lock(board_mutex[pos1]);
  int id1 = PureBoardPos(pos1);
  int c = col - 1;
  //int index = (c * pure_board_max + pos1) * 0xffff + hash_last;
  int index = (c * pure_board_max + id1);
  uint16_t count = min(tgr_num, tgr1_count[index]);
  for (int i = 0; i < count; i++) {
    int idx = index * tgr_num + i;
    if (tgr1_hash[idx] == hash
      && tgr1_last[idx] == hash_last
      && tgr1_lib[idx] == lib
      && tgr1_lib_last[idx] == lib_last) {
      tgr1[idx] = val;
      return;
    }
  }
  uint16_t n = tgr1_count[index] ++;
  n = n % tgr_num;
  int idx = index * tgr_num + n;
  tgr1[idx] = val;
  tgr1_hash[idx] = hash;
  tgr1_last[idx] = hash_last;
  tgr1_lib[idx] = lib;
  tgr1_lib_last[idx] = lib_last;
}

int
LGR::getTGR1(int col, int pos1, const game_info_t* game)
{
  if (pos1 == PASS) return PASS;
  lock_guard<mutex> lock(board_mutex[pos1]);
  int id1 = PureBoardPos(pos1);
  int c = col - 1;
  uint32_t hash_last = MD2(game->pat, pos1);
  uint16_t lib_last = GetLib(game, pos1);
  //int index = (c * pure_board_max + pos1) * 0xffff + hash_last;
  int index = c * pure_board_max + id1;
  uint16_t count = min(tgr_num, tgr1_count[index]);
  for (int i = 0; i < count; i++) {
    int idx = index * tgr_num + i;
    int pos = tgr1[idx];
    if (pos == PASS)
      continue;
    if (tgr1_last[idx] != hash_last)
      continue;
    if (tgr1_lib_last[idx] != lib_last)
      continue;
    uint32_t hash = MD2(game->pat, pos);
    //PrintBoard(game);
    //cerr << FormatMove(pos1) << endl;
    //DisplayInputMD2(hash_last);
    //cerr << FormatMove(pos) << endl;
    //DisplayInputMD2(hash);
    //DisplayInputMD2(tgr1_hash[index * tgr_num + i]);
    if (tgr1_hash[idx] == hash) {
      uint16_t lib = GetLib(game, pos);
      if (tgr1_lib[idx] == lib) {
        //cerr << "HIT" << endl;
        //DisplayInputMD2(hash);
        //cerr << lib << endl;
        return pos;
      } else {
        //cerr << "LIB MISMATCH" << endl;
        //DisplayInputMD2(hash);
        //cerr << lib << endl;
      }
    }
    //cerr << "MISS" << endl;
  }
  return PASS;
}

int
LGR::getLGRF1(int col, int pos1, const game_info_t* game)
{
  if (pos1 == PASS) return PASS;
  lock_guard<mutex> lock(board_mutex[pos1]);
  int id1 = PureBoardPos(pos1);
  int c = col - 1;
  uint32_t hash_last = MD2(game->pat, pos1);
  if (lgrf1_last[c * pure_board_max + id1] != hash_last)
    return PASS;
  return lgrf1[c * pure_board_max + id1];
}

void
LGR::setLGRF1(int col, int pos1, int pos, uint32_t hash2)
{
  lock_guard<mutex> lock(board_mutex[pos1]);
  int id1 = PureBoardPos(pos1);
  int c = col - 1;
  lgrf1[c * pure_board_max + id1] = pos;
  lgrf1_last[c * pure_board_max + id1] = hash2;
}

void
LGR::clearLGRF1(int col, int pos1)
{
  this->setLGRF1(col, pos1, PASS, 0);
}

int
LGR::getLGRF2(int col, int pos1, int pos2)
{
  if (pos1 == PASS) return PASS;
  lock_guard<mutex> lock(board_mutex[pos1]);
  pos1 = PureBoardPos(pos1);
  pos2 = PureBoardPos(pos2);
  int c = col - 1;
  return lgrf2[(c * pure_board_max + pos1) * pure_board_max + pos2];
}

void
LGR::setLGRF2(int col, int pos1, int pos2, int val)
{
  pos1 = PureBoardPos(pos1);
  pos2 = PureBoardPos(pos2);
  int c = col - 1;
  lgrf2[(c * pure_board_max + pos1) * pure_board_max + pos2] = val;
}

bool
LGR::hasLGRF2(int col, int pos1, int pos2)
{
  return (this->getLGRF2(col, pos1, pos2) != PASS);
}

void
LGR::clearLGRF2(int col, int pos1, int pos2)
{
  this->setLGRF2(col, pos1, pos2, PASS);
}

void
LGR::update(game_info_t* game, int start, int win, const LGRContext& ctx)
{
  if (tgr1_rate > 0) {
    for (int i = 1; i < game->moves; i++) {
      int c = game->record[i].color;
      int mp = game->record[i].pos;
      unsigned int h = ctx.hash[i];
      uint16_t lib = ctx.lib[i];
      int pos1 = game->record[i - 1].pos;
      unsigned int h1 = ctx.hash_last[i];
      uint16_t lib1 = ctx.lib_last[i];

      bool iswin = (c == win);
      if (mp == PASS || pos1 == PASS)
	continue;
      if (h == 0 || h1 == 0)
	continue;
      if (i < start) {
	this->setTGR1(c, pos1, mp, h1, h, lib1, lib);
      } else {
	if (iswin) {
	  //this->setTGR1(c, p1, mp, h1, h);
	}
      }
    }
  }

  if (lgrf1_rate > 0) {
    for (int i = 1; i < game->moves; i++) {
      int c = game->record[i].color;
      int mp = game->record[i].pos;
      //unsigned int h = ctx.hash[i];
      int p1 = game->record[i - 1].pos;
      uint32_t h1 = ctx.hash_last[i];

      bool iswin = (c == win);
      if (mp == PASS || p1 == PASS)
	continue;
      if (iswin) {
	//cerr << "adding LGRF1: " << FormatMove(p1) << " -> " << FormatMove(mp) << endl;
	this->setLGRF1(c, p1, mp, h1);
      } else {
	//cerr << "forgetting LGRF1: " << FormatMove(p1) << endl;
	this->clearLGRF1(c, p1);
      }
#if 0
      if (iswin)
      {
	//cerr << "adding LGRF1: " << FormatMove(p1) << " -> " << FormatMove(mp) << endl;
	this->setLGRF1(c, p1, mp, h1, h);
      } else {
	//cerr << "forgetting LGRF1: " << FormatMove(p1) << endl;
	//if (i > strat) this->clearLGRF1(c, p1);
	this->setLGRF1(c, p1, PASS, h1, 0);
      }
#endif
    }
  }

  if (use_lgrf2) {
    for (int i = 2; i < game->moves; i++) {
      int c = game->record[i].color;
      int mp = game->record[i].pos;
      int p1 = game->record[i - 2].pos;
      int p2 = game->record[i - 1].pos;

      if (mp == PASS || p1 == PASS || p2 == PASS)
        break;

      bool iswin = (c == win);

      if (iswin) {
	//cerr << "adding LGRF2: " << FormatMove(p1) << " -> " << FormatMove(p2) << " -> " << FormatMove(mp) << endl;
	this->setLGRF2(c, p1, p2, mp);
	//this->setLGRF1o(c, p1, mp);
      } else {
	//cerr << "forgetting LGRF2: " << FormatMove(p1) << " -> " << FormatMove(p2) << endl;
	if (i > start)
	  this->clearLGRF2(c, p1, p2);
	//this->clearLGRF1o(c, p1);
      }
    }
  }
}

void
LGRContext::store(const game_info_t* game, int pos) {
  if (pos == PASS) {
    hash[game->moves] = 0;
    hash_last[game->moves] = 0;
    lib[game->moves] = 0;
    lib_last[game->moves] = 0;
  } else {
    int gm1 = game->record[game->moves - 1].pos;
    hash[game->moves] = MD2(game->pat, pos);
    hash_last[game->moves] = MD2(game->pat, gm1);
    lib[game->moves] = GetLib(game, pos);
    lib_last[game->moves] = GetLib(game, gm1);
  }
}
