#pragma once

#include <vector>

#include "GoBoard.h"

struct book_move_t {
  int pos;
  int win;
  int move_count;
};


void LoadOpeningBook(int size);

const std::vector<book_move_t>* LookupOpeningBook(const game_info_t * game);
