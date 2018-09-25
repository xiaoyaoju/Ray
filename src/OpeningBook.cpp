#include "OpeningBook.h"
#include "Point.h"
#include "Message.h"
#include "UctRating.h"
#include "Utility.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <map>

using namespace std;

struct book_element_t {
  int hash;
  vector<int> path;
  vector<book_move_t> moves;
};

static string kifu_dir = "kifu";
static multimap<unsigned long long, book_element_t> opening_books;

static int path_hash(const vector<int> &path)
{
  int h = 0;
  for (int p : path)
    h = h * 31 + p;
  return h;
}

static int path_hash(const game_info_t *game)
{
  int h = 0;
  for (int i = 1; i < game->moves; i++)
    h = h * 31 + game->record[i].pos;
  return h;
}

void
LoadOpeningBook(int size)
{
  string uct_parameters_path = uct_params_path;
  string path;

#if defined (_WIN32)
  uct_parameters_path += '\\';
#else
  uct_parameters_path += '/';
#endif

  path = uct_parameters_path + "book.txt";

  opening_books.clear();

  ifstream in{ path };
  std::string line;
  while (getline(in, line)) {
    stringstream ss{ line };
    int s;
    ss >> s;
    if (s != size)
      continue;
    book_element_t e;
    while (true) {
      string move;
      ss >> move;
      if (move == "|")
        break;
      e.path.push_back(StringToInteger(move.c_str()));
    }
    e.hash = path_hash(e.path);

    while (!ss.eof()) {
      string move;
      ss >> move;
      if (move.length() < 2) {
        break;
      }
      book_move_t m;
      m.pos = StringToInteger(move.c_str());
      ss >> m.win >> m.move_count;
      e.moves.push_back(m);
    }
    /*
    cerr << s << " ";
    for (int pos : e.path)
      cerr << FormatMove(pos) << "->";
    cerr << endl;
    cerr << "\t";
    for (book_move_t &m : e.moves)
      cerr << FormatMove(m.pos) << "(" << m.win << "/" << m.move_count << ") ";
    cerr << endl;
    */
    opening_books.emplace(e.hash, e);
  }

  //cerr << "OK " << opening_books.size() << endl;
}

const vector<book_move_t>*
LookupOpeningBook(const game_info_t * game)
{
  int h = path_hash(game);
  auto r = opening_books.equal_range(h);
  for (auto it = r.first; it != r.second; it++) {
    auto& e = it->second;
    if (e.hash != h)
      continue;
    bool match = true;
    if (e.path.size() != game->moves - 1) {
      match = false;
    } else {
      for (int i = 1; i < game->moves; i++) {
        if (e.path[i - 1] != game->record[i].pos) {
          match = false;
          break;
        }
      }
    }
    if (!match) {
      cerr << "Hash Collision" << endl;
      continue;
    }
    /*
    for (book_move_t &m : e.moves)
      cerr << FormatMove(m.pos) << "(" << m.win << "/" << m.move_count << ") ";
    cerr << endl;
    */
    return &e.moves;
  }

  return nullptr;
}
