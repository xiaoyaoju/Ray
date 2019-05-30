#pragma once

#include <map>
#include <vector>
#include <iostream>

#include "GoBoard.h"

struct book_move_t {
  int pos;
  int win;
  int move_count;
};

struct book_element_t {
  int hash;
  std::vector<int> path;
  std::vector<book_move_t> moves;
};

class OpeningBook {
public:
  void load(int size);
  void load(int size, std::istream& in);
  void save(std::ostream& out);
  const std::vector<book_move_t>* lookup(const game_info_t * game);
private:
  std::multimap<unsigned long long, book_element_t> books;
};

void
ThinkOpeningBook();
