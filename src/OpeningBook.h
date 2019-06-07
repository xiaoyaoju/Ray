#pragma once

#include <map>
#include <vector>
#include <iostream>

#include "GoBoard.h"

struct book_move_t {
  int pos;
  int win;
  int move_count;
  double value;
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
  std::pair<const std::vector<book_move_t>*, int> lookup(const game_info_t * game);
  std::pair<const std::vector<book_move_t>*, int> lookup(const std::vector<int> &path);

  std::multimap<unsigned long long, std::pair<int, int>> books;
  std::vector<book_element_t> elements;
  int move_count;
  int value_move_count;
};

void
ThinkOpeningBook();
