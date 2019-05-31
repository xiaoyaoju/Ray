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

static int PathHash(const vector<int> &path)
{
  int h = 0;
  for (int p : path)
    h = h * 31 + p;
  return h;
}

static int PathHash(const game_info_t *game)
{
  int h = 0;
  for (int i = 1; i < game->moves; i++)
    h = h * 31 + game->record[i].pos;
  return h;
}

static vector<int>
TransformPath(const vector<int> &path, int trans)
{
  vector<int> r;
  for (int pos : path)
    r.push_back(TransformMove(pos, trans));
  return r;
}

void
OpeningBook::load(int size)
{
  string uct_parameters_path = uct_params_path;
  string path;

#if defined (_WIN32)
  uct_parameters_path += '\\';
#else
  uct_parameters_path += '/';
#endif

  path = uct_parameters_path + "book.txt";

  ifstream in{ path };
  load(size, in);
}

void
OpeningBook::load(int size, istream& in)
{
  books.clear();
  elements.clear();

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
    e.hash = PathHash(e.path);

    bool exists = false;
    int thash[8];
    std::vector<int> tpath[8];
    for (int i = 0; i < 8; i++) {
      tpath[i] = TransformPath(e.path, i);
      thash[i] = PathHash(tpath[i]);
      if (lookup(tpath[i]).first != nullptr) {
        exists = true;
      }
    }
    if (exists) {
      cerr << "skip" << endl;
      continue;
    }

    while (!ss.eof()) {
      string move;
      ss >> move;
      if (move.length() < 2) {
        break;
      }
      book_move_t m;
      m.pos = StringToInteger(move.c_str());
      ss >> m.win >> m.move_count;
      ss >> m.value;
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
    int id = elements.size();
    elements.push_back(e);
    for (int i = 0; i < 8; i++) {
      books.emplace(thash[i], make_pair(id, i));
    }
  }

  if (GetDebugMessageMode())
    cerr << "OK " << elements.size() << " " << books.size() << endl;
}

std::pair<const std::vector<book_move_t>*, int>
OpeningBook::lookup(const vector<int> &path)
{
  int h = PathHash(path);
  auto r = books.equal_range(h);
  for (auto it = r.first; it != r.second; it++) {
    int id = it->second.first;
    int trans = it->second.second;
    auto &e = elements[id];
    bool match = true;
    if (e.path.size() != path.size()) {
      match = false;
    } else {
      for (int i = 0; i < path.size(); i++) {
        if (TransformMove(e.path[i], trans) != path[i]) {
          match = false;
          break;
        }
      }
    }
    if (!match) {
      if (GetDebugMessageMode() && GetVerbose())
        cerr << "Hash Collision" << endl;
      continue;
    }
    return make_pair(&e.moves, trans);
  }

  return make_pair(nullptr, 0);
}

std::pair<const std::vector<book_move_t>*, int>
OpeningBook::lookup(const game_info_t * game)
{
  int h = PathHash(game);
  auto r = books.equal_range(h);
  for (auto it = r.first; it != r.second; it++) {
    int id = it->second.first;
    int trans = it->second.second;
    auto &e = elements[id];
    bool match = true;
    if (e.path.size() != game->moves - 1) {
      match = false;
    } else {
      for (int i = 1; i < game->moves; i++) {
        if (TransformMove(e.path[i - 1], trans) != game->record[i].pos) {
          match = false;
          break;
        }
      }
    }
    if (!match) {
      if (GetDebugMessageMode() && GetVerbose())
        cerr << "Hash Collision" << endl;
      continue;
    }
    /*
    for (book_move_t &m : e.moves)
      cerr << FormatMove(m.pos) << "(" << m.win << "/" << m.move_count << ") ";
    cerr << endl;
    */
    return make_pair(&e.moves, trans);
  }

  return make_pair(nullptr, 0);
}

void
OpeningBook::save(std::ostream& out)
{
  for (auto& it = books.begin(); it != books.end(); it++) {
    auto& r = it->second;
    if (r.second != 0)
      continue;
    auto& elm = elements[r.first];
    out << pure_board_size;
    for (int pos : elm.path) {
      out << " " << FormatMove(pos);
    }
    out << " |";

    for (auto m : elm.moves) {
      out << " " << FormatMove(m.pos)
          << " " << m.win
          << " " << m.move_count
          << " " << m.value;
    }
    out << endl;
  }
}

/*
void
CreateTransposePos()
{
  for (auto& it = books.begin(); it != books.end(); it++) {
    auto& elm = it->second;
    for (int pos : elm.path) {
      out << " " << FormatMove(pos);
    }
    out << " |";
  }
}
*/

void
ThinkOpeningBook()
{
  OpeningBook book;
  book.load(pure_board_size);

#if 1
  vector<int> indicies;
  for (int i = 0; i < book.elements.size(); i++)
    indicies.push_back(i);
  sort(indicies.begin(), indicies.end(), [&](const int a, const int b) {
      return book.elements[a].path.size() > book.elements[b].path.size();
    });
  for (int i = 0; i < book.elements.size(); i++) {
    auto& elm = book.elements[indicies[i]];
    cerr << "# THINK " << elm.path.size()
      << " " << i << "/" << book.elements.size()
      << endl;

    game_info_t *game = AllocateGame();
    InitializeBoard(game);

    for (auto& m : elm.moves) {
      cerr << "## " << FormatMove(m.pos) << endl;
      if (m.value >= 0.0 && m.value <= 1.0) {
        cerr << m.value << endl;
        continue;
      }
      ClearBoard(game);
      int color = S_BLACK;
      for (int pos : elm.path) {
        PutStone(game, pos, color);
        color = FLIP_COLOR(color);
      }
      PutStone(game, m.pos, color);
      color = FLIP_COLOR(color);

      PrintBoard(game);
      UctSearchGenmove(game, color);
      auto root = &uct_node[current_root];
      m.value = 1 - (double)root->value_win / root->value_move_count;

      cerr << " -->"
           << " V:" << (100.0 * m.value)
           << " G:" << (100.0 * m.win / m.move_count)
           << " (" <<  m.move_count << ")"
           << endl;
    }

    FreeGame(game);

    ofstream out("book_out.txt");
    book.save(out);
  }
#endif

  ofstream out("book_out.txt");
  book.save(out);
}
