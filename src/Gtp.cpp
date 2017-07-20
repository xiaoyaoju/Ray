#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cctype>
#include <iostream>
#include <memory>
#include <fstream>
#include <vector>
#include <random>
#include <sstream>
#include <iomanip>

#include "DynamicKomi.h"
#include "Gtp.h"
#include "GoBoard.h"
#include "Nakade.h"
#include "UctSearch.h"
#include "UctRating.h"
#include "Message.h"
#include "Point.h"
#include "Rating.h"
#include "Simulation.h"
#include "UctSearch.h"
#include "Utility.h"
#include "ZobristHash.h"
#include "MoveCache.h"

using namespace std;

// gtpの出力用関数
static void GTP_response( const char *res, bool success );
// boardsizeコマンドを処理
static void GTP_boardsize( void );
// clearboardコマンドを処理
static void GTP_clearboard( void );
// nameコマンドを処理
static void GTP_name( void );
// protocolversionコマンドを処理
static void GTP_protocolversion( void );
// genmoveコマンドを処理
static void GTP_genmove( void );
// playコマンドを処理
static void GTP_play( void );
// knowncommandコマンドを処理
static void GTP_knowncommand( void );
// listcommandsコマンドを処理
static void GTP_listcommands( void );
// quitコマンドを処理
static void GTP_quit( void );
// komiコマンドを処理
static void GTP_komi( void );
// getkomiコマンドを処理
static void GTP_getkomi( void );
// finalscoreコマンドを処理
static void GTP_finalscore( void );
// timesettingsコマンドを処理
static void GTP_timesettings( void );
// timeleftコマンドを処理
static void GTP_timeleft( void );
// versionコマンドを処理
static void GTP_version( void );
// showboardコマンドを処理
static void GTP_showboard( void );
// kgs-genmove_cleanupコマンドを処理
static void GTP_kgs_genmove_cleanup( void );
// final_status_listコマンドを処理
static void GTP_final_status_list( void );
// set_free_handicapコマンドを処理
static void GTP_set_free_handicap( void );
// fixed_handicapコマンドを処理
static void GTP_fixed_handicap( void );
// GOGUI
static void GTP_gogui_analyze_commands( void );
//
static void GTP_ray_toggle_live_best_sequence();
//
static void GTP_ray_best_sequence();
//
static void GTP_ray_param();
//
static void GTP_ray_stat();
//
static void GTP_features_planes_file(void);
//
static void GTP_features_clear(void);
//
static void GTP_features_store(void);
//
static void GTP_stat(void);
//
static void GTP_stat_po(void);


const GTP_command_t gtpcmd[GTP_COMMAND_NUM] = {
  { "boardsize",           GTP_boardsize           },
  { "clear_board",         GTP_clearboard          },
  { "name",                GTP_name                },
  { "protocol_version",    GTP_protocolversion     },
  { "genmove",             GTP_genmove             }, 
  { "play",                GTP_play                },
  { "known_command",       GTP_knowncommand        },
  { "list_commands",       GTP_listcommands        },
  { "quit",                GTP_quit                },
  { "komi",                GTP_komi                },
  { "get_komi",            GTP_getkomi             },
  { "final_score",         GTP_finalscore          },
  { "time_settings",       GTP_timesettings        },
  { "time_left",           GTP_timeleft            },
  { "version",             GTP_version             },
  { "genmove_black",       GTP_genmove             },
  { "genmove_white",       GTP_genmove             },
  { "black",               GTP_play                },
  { "white",               GTP_play                },
  { "showboard",           GTP_showboard           },
  { "final_status_list",   GTP_final_status_list   },
  { "fixed_handicap",      GTP_fixed_handicap      },
  { "place_free_handicap", GTP_fixed_handicap      },
  { "set_free_handicap",   GTP_set_free_handicap   },
  { "kgs-genmove_cleanup", GTP_kgs_genmove_cleanup },
  { "gogui-analyze_commands", GTP_gogui_analyze_commands },
  { "ray-toggle_live_best_sequence", GTP_ray_toggle_live_best_sequence },
  { "ray-best_sequence", GTP_ray_best_sequence },
  { "ray-param", GTP_ray_param },
  { "ray-stat", GTP_ray_stat },
  { "_clear", GTP_features_clear },
  { "_store", GTP_features_store },
  { "_dump", GTP_features_planes_file },
  { "_stat", GTP_stat_po },
};

static char input[BUF_SIZE], input_copy[BUF_SIZE];
static char *next_token;

char brank[] = "";
char err_command[] = "? unknown command";
char err_genmove[] = "gemmove color";
char err_play[] = "play color point";
char err_komi[] = "komi float";

static int player_color = 0;

static game_info_t *game;

static game_info_t *game_prev;
static game_info_t *store_game;
static uct_node_t store_node;
static double store_winning_percentage;

static unique_ptr<ofstream> stream_ptr;
static bool sim_move = false;


////////////////////////
//  void SetSimMove() //
////////////////////////
void
SetSimMove( bool flag )
{
  sim_move = flag;
}

///////////////////////
//  void GTP_main()  //
///////////////////////
void
GTP_main( void )
{
  game = AllocateGame();
  InitializeBoard(game);

  game_prev = AllocateGame();
  InitializeBoard(game_prev);
  store_game = AllocateGame();
  InitializeBoard(store_game);

  while (true) {
    if (fgets(input, sizeof(input), stdin) == NULL) {
      if (feof(stdin))
        break;
      continue;
    }
    char *command;
    bool nocommand = true;

    STRCPY(input_copy, BUF_SIZE, input);
    command = STRTOK(input, DELIM, &next_token);
    CHOMP(command);

    for (int i = 0; i < GTP_COMMAND_NUM; i++) {
      if (!strcmp(command, gtpcmd[i].command)) {
	StopPondering();
	(*gtpcmd[i].function)();
	nocommand = false;
	break;
      }
    }

    if (nocommand) {
      cout << err_command << endl << endl;
    }

    fflush(stdin);
    fflush(stdout);
  }
  cerr << "EXIT" << endl;
}


/////////////////////////////////////////////////
//  void GTP_response(char *res, int success)  //
/////////////////////////////////////////////////
static void
GTP_response( const char *res, bool success )
{
  if (success){
    cout << "= " << res << endl << endl;
  } else {
    if (res != NULL) {
      cerr << res << endl;
    }
    cout << "?" << endl << endl;
  }
}


/////////////////////////////
//　 void GTP_boardsize()  //
/////////////////////////////
static void
GTP_boardsize( void )
{
  char *command;
  int size;
  char buf[1024];
  
  command = STRTOK(NULL, DELIM, &next_token);

#if defined (_WIN32)
  sscanf_s(command, "%d", &size);
  sprintf_s(buf, 1024, " ");
#else
  sscanf(command, "%d", &size);
  snprintf(buf, 1024, " ");
#endif

  if (pure_board_size != size &&
      size <= PURE_BOARD_SIZE && size > 0) {
    SetBoardSize(size);
    SetParameter();
    SetNeighbor();
    InitializeNakadeHash();
  }

  FreeGame(game);
  game = AllocateGame();
  InitializeBoard(game);
  InitializeSearchSetting();
  InitializeUctHash();

  GTP_response(brank, true);
}
  

/////////////////////////////
//  void GTP_clearboard()  //
/////////////////////////////
static void
GTP_clearboard( void )
{
  player_color = 0;
  SetHandicapNum(0);
  FreeGame(game);
  game = AllocateGame();
  InitializeBoard(game);
  InitializeSearchSetting();
  InitializeUctHash();

  GTP_response(brank, true);
}
  

///////////////////////
//  void GTP_name()  //
///////////////////////
static void
GTP_name( void )
{
  GTP_response(PROGRAM_NAME, true);
}


//////////////////////////////////
//  void GTP_protocolversion()  //
//////////////////////////////////
static void
GTP_protocolversion( void )
{
  GTP_response(PROTOCOL_VERSION, true);
}


//////////////////////////
//  void GTP_genmove()  //
//////////////////////////
static void
GTP_genmove( void )
{
  char *command;
  char c;
  char pos[10];
  int color;
  int point = PASS;
  
  command = STRTOK(input_copy, DELIM, &next_token);
  
  CHOMP(command);
  if (!strcmp("genmove_black", command)) {
    color = S_BLACK;
  } else if (!strcmp("genmove_white", command)) {
    color = S_WHITE;
  } else {
    command = STRTOK(NULL, DELIM, &next_token);
    if (command == NULL){
      GTP_response(err_genmove, true);
      return;
    }
    CHOMP(command);
    c = (char)tolower((int)command[0]);
    if (c == 'w') {
      color = S_WHITE;
    } else if (c == 'b') {
      color = S_BLACK;
    } else {
      GTP_response(err_genmove, true);
      return;
    }
  }

  player_color = color;

  if (sim_move)
    point = SimulationGenmove(game, color);
  else
    point = UctSearchGenmove(game, color);
  if (point != RESIGN) {
    PutStone(game, point, color);
  }
  
  IntegerToString(point, pos);
  
  GTP_response(pos, true);

  UctSearchPondering(game, FLIP_COLOR(color));
}


///////////////////////
//  void GTP_play()  //
///////////////////////
static void
GTP_play( void )
{
  char *command;
  char c;
  int color, pos = 0;
  
  command = STRTOK(input_copy, DELIM, &next_token);

  if (!strcmp("black", command)){
    color = S_BLACK;
  } else if (!strcmp("white", command)){
    color = S_WHITE;
  } else{
    command = STRTOK(NULL, DELIM, &next_token);
    if (command == NULL){
      GTP_response(err_play, false);
      return;
    }
    CHOMP(command);
    c = (char)tolower((int)command[0]);
    if (c == 'w') {
      color = S_WHITE;
    } else{
      color = S_BLACK;
    }
  }

  command = STRTOK(NULL, DELIM, &next_token);
  
  CHOMP(command);
  if (command == NULL){
    GTP_response(err_play, false);
    return;
  } else {
    pos = StringToInteger(command);
  }

  CopyGame(game_prev, game);
  if (pos != RESIGN) {
    PutStone(game, pos, color);
  }
  //PrintBoard(game);
  
  GTP_response(brank, true);
}
 

/////////////////////////////
// void GTP_knowncommand() //
/////////////////////////////
static void
GTP_knowncommand( void )
{
  char *command;
  
  command = STRTOK(NULL, DELIM, &next_token);
  
  if (command == NULL){
    GTP_response("known_command command", false);
    return;
  }
  CHOMP(command);
  for (int i = 0; i < GTP_COMMAND_NUM; i++){
    if (!strcmp(command, gtpcmd[i].command)) {
      GTP_response("true", true);
      return;
    }
  }
  GTP_response("false", false);
}
 
 
///////////////////////////////
//  void GTP_listcommands()  //
///////////////////////////////
static void
GTP_listcommands( void )
{
  char list[2048];
  int i;

  i = 0;
  list[i++] = '\n';
  for (int j = 0; j < GTP_COMMAND_NUM; j++) {
    for (unsigned int k = 0; k < strlen(gtpcmd[j].command); k++){
      list[i++] = gtpcmd[j].command[k];
    }
    list[i++] = '\n';
  }
  list[i++] = '\0';

  GTP_response(list, true);
}
 
 
//////////////////////
// void GTP_quit()  //
//////////////////////
static void
GTP_quit( void )
{
  FinalizeUctSearch();
  GTP_response(brank, true);
  exit(0);
}
 
 
///////////////////////
//  void GTP_komi()  //
///////////////////////
static void
GTP_komi( void )
{
  char* c_komi;
  
  c_komi = STRTOK(NULL, DELIM, &next_token);

  if (c_komi != NULL) {
    SetKomi(atof(c_komi));
    PrintKomiValue();
    GTP_response(brank, true);
  } else {
    GTP_response(err_komi, false);
  }
}
 

//////////////////////////
//  void GTP_getkomi()  //
//////////////////////////
static void
GTP_getkomi( void )
{
  char buf[256];
  
#if defined(_WIN32)
  sprintf_s(buf, 4, "%lf", komi[0]);
#else
  snprintf(buf, 4, "%lf", komi[0]);
#endif
  GTP_response(buf, true);
}


/////////////////////////////
//  void GTP_finalscore()  //
/////////////////////////////
static void
GTP_finalscore( void )
{
  char buf[10];
  double score = 0;
  
  score = UctAnalyze(game, S_BLACK) - komi[0];

#if defined(_WIN32)  
  if (score > 0) {
    sprintf_s(buf, 10, "B+%.1lf", score);
  } else {
    sprintf_s(buf, 10, "W+%.1lf", abs(score));
  }
#else
  if (score > 0) {
    snprintf(buf, 10, "B+%.1lf", score);
  } else {
    snprintf(buf, 10, "W+%.1lf", abs(score));
  }
#endif

  GTP_response(buf, true);
}
 

///////////////////////////////
//  void GTP_timesettings()  //
///////////////////////////////
static void
GTP_timesettings( void )
{
  char *str1, *str2, *str3;
  double main_time, byoyomi, stone;

  str1 = STRTOK(NULL, DELIM, &next_token);
  str2 = STRTOK(NULL, DELIM, &next_token);
  str3 = STRTOK(NULL, DELIM, &next_token);

  main_time = atoi(str1);
  byoyomi = atoi(str2);
  stone = atoi(str3);

  cerr << main_time << "," << byoyomi << "," << stone << endl;

  SetTimeSettings(main_time, byoyomi, stone);
  InitializeSearchSetting();
  
  GTP_response(brank, true);
}


///////////////////////////
//  void GTP_timeleft()  //
///////////////////////////
static void
GTP_timeleft( void )
{
  char *str1, *str2;

  str1 = STRTOK(NULL, DELIM, &next_token);
  str2 = STRTOK(NULL, DELIM, &next_token);
  
  if (str1[0] == 'B' || str1[0] == 'b'){
    remaining_time[S_BLACK] = atof(str2);
  } else if (str1[0] == 'W' || str1[0] == 'w'){
    remaining_time[S_WHITE] = atof(str2);
  }
  
  fprintf(stderr, "%f\n", remaining_time[S_BLACK]);
  fprintf(stderr, "%f\n", remaining_time[S_WHITE]);
  GTP_response(brank, true);
}


//////////////////////////
//  void GTP_version()  //
//////////////////////////
static void
GTP_version( void )
{
  GTP_response(PROGRAM_VERSION, true);
}
 
 
////////////////////////////
//  void GTP_showboard()  //
////////////////////////////
static void
GTP_showboard( void )
{
  PrintBoard(game);
  GTP_response(brank, true);
}


/////////////////////////////////
//  void GTP_fixed_handicap()  //
/////////////////////////////////
static void
GTP_fixed_handicap( void )
{
  char *command;
  int num;
  char buf[1024];
  char pos[5];
  int handicap[9];
  const int place_index[8][9] = {
    {2, 6},
    {0, 2, 6},
    {0, 2, 6, 8},
    {0, 2, 4, 6, 8},
    {0, 2, 3, 5, 6, 8},
    {0, 2, 3, 4, 5, 6, 8},
    {0, 1, 2, 3, 5, 6, 7, 8},
    {0, 1, 2, 3, 4, 5, 6, 7, 8},
  };
  
  command = STRTOK(NULL, DELIM, &next_token);
  
#if defined (_WIN32)
  sscanf_s(command, "%d", &num);
  sprintf_s(buf, 1024, " ");
#else
  sscanf(command, "%d", &num);
  snprintf(buf, 1024, " ");
#endif

  if (num < 2 || 9 < num) {
    GTP_response(brank, false);
    return ;
  }

  handicap[0] = POS(board_start +  3, board_start +  3);
  handicap[1] = POS(board_start +  9, board_start +  3);
  handicap[2] = POS(board_start + 15, board_start +  3);
  handicap[3] = POS(board_start +  3, board_start +  9);
  handicap[4] = POS(board_start +  9, board_start +  9);
  handicap[5] = POS(board_start + 15, board_start +  9);
  handicap[6] = POS(board_start +  3, board_start + 15);
  handicap[7] = POS(board_start +  9, board_start + 15);
  handicap[8] = POS(board_start + 15, board_start + 15);

  for (int i = 0; i < num; i++) {
    PutStone(game, handicap[place_index[num - 2][i]], S_BLACK);
#if defined (_WIN32)
    sprintf_s(pos, 5, "%c%d ", GOGUI_X(handicap[place_index[num - 2][i]]), GOGUI_Y(handicap[place_index[num - 2][i]]));
    strcat_s(buf, 1024, pos);
#else
    snprintf(pos, 5, "%c%d ", GOGUI_X(handicap[place_index[num - 2][i]]), GOGUI_Y(handicap[place_index[num - 2][i]]));
    strncat(buf, pos, 5);
#endif
  }
  
  SetKomi(0.5);
  SetHandicapNum(num);
  GTP_response(buf, true);
}


////////////////////////////////////
//  void GTP_set_free_handicap()  //
////////////////////////////////////
static void
GTP_set_free_handicap( void )
{
  char *command;
  int pos, num = 0;
  
  while (1){
    command = STRTOK(NULL, DELIM, &next_token);

    if (command == NULL){
      SetHandicapNum(num);
      SetKomi(0.5);
      GTP_response(brank, true);
      return;
    }
    
    pos = StringToInteger(command);
    
    if (pos > 0 && pos < board_max && IsLegal(game, pos, S_BLACK)) {
      PutStone(game, pos, S_BLACK);
      num++;
    }
  }
}


////////////////////////////////////
//  void GTP_final_status_list()  //
////////////////////////////////////
static void
GTP_final_status_list( void )
{
  char dead[2048] = { 0 };
  char pos[5];
  int owner[BOARD_MAX]; 
  char *command;
  
  OwnerCopy(owner);
  
  command = STRTOK(NULL, DELIM, &next_token);

  CHOMP(command);
  
  if (!strcmp(command, "dead")){
    for (int y = board_start; y <= board_end; y++) {
      for (int x = board_start; x <= board_end; x++) {
	if ((game->board[POS(x, y)] == player_color && owner[POS(x, y)] <= 30) ||
	    (game->board[POS(x, y)] == FLIP_COLOR(player_color) && owner[POS(x, y)] >= 70)) {
#if defined (_WIN32)
	  sprintf_s(pos, 5, "%c%d ", GOGUI_X(POS(x, y)), GOGUI_Y(POS(x, y)));
	  strcat_s(dead, 2048, pos);
#else
	  snprintf(pos, 5, "%c%d ", GOGUI_X(POS(x, y)), GOGUI_Y(POS(x, y)));
	  strncat(dead, pos, 5);
#endif
	}
      }
    }
  } else if (!strcmp(command, "alive")){
    for (int y = board_start; y <= board_end; y++) {
      for (int x = board_start; x <= board_end; x++) {
	if ((game->board[POS(x, y)] == player_color && owner[POS(x, y)] >= 70) ||
	    (game->board[POS(x, y)] == FLIP_COLOR(player_color) && owner[POS(x, y)] <= 30)) {
#if defined (_WIN32)
	  sprintf_s(pos, 5, "%c%d ", GOGUI_X(POS(x, y)), GOGUI_Y(POS(x, y)));
	  strcat_s(dead, 2048, pos);
#else
	  snprintf(pos, 5, "%c%d ", GOGUI_X(POS(x, y)), GOGUI_Y(POS(x, y)));
	  strncat(dead, pos, 5);
#endif
	}
      }
    }
  }
  
  GTP_response(dead, true);
}


//////////////////////////////////////
//  void GTP_kgs_genmove_cleanup()  //
//////////////////////////////////////
static void
GTP_kgs_genmove_cleanup( void )
{
  char *command;
  char c;
  char pos[10];
  int color;
  int point = PASS;
  
  command = STRTOK(input_copy, DELIM, &next_token);
  
  CHOMP(command);
  if (!strcmp("genmove_black", command)) {
    color = S_BLACK;
  } else if (!strcmp("genmove_white", command)) {
    color = S_WHITE;
  } else {
    command = STRTOK(NULL, DELIM, &next_token);
    if (command == NULL){
      GTP_response(err_genmove, false);
      return;
    }
    CHOMP(command);
    c = (char)tolower((int)command[0]);
    if (c == 'w') {
      color = S_WHITE;
    } else if (c == 'b') {
      color = S_BLACK;
    } else {
      GTP_response(err_genmove, false);
      return;
    }
  }
  
  player_color = color;
  
  point = UctSearchGenmoveCleanUp(game, color);
  if (point != RESIGN) {
    PutStone(game, point, color);
  }
  
  IntegerToString(point, pos);
  
  GTP_response(pos, true);
}

/////////////////////////////////////////
//  void GTP_gogui_analyze_commands()  //
/////////////////////////////////////////
static void
GTP_gogui_analyze_commands()
{
  GTP_response(
    "none/Togle Live Best Sequence/ray-toggle_live_best_sequence\n"
    "gfx/Print Best Sequence/ray-best_sequence %m\n"
    "hpstring/Print Moves/ray-stat %m\n"
    "",
    true);
}

static void
GTP_ray_toggle_live_best_sequence()
{
  ToggleLiveBestSequence();
  GTP_response(brank, true);
}

////////////////////////////////////
//  void GTP_ray_best_sequence()  //
////////////////////////////////////
static void
GTP_ray_best_sequence()
{
  char *command;
  char c;
  char pos[10];
  int color;
  int point = PASS;

  command = STRTOK(input_copy, DELIM, &next_token);

  CHOMP(command);
  command = STRTOK(NULL, DELIM, &next_token);
  if (command == NULL) {
    GTP_response(err_genmove, true);
    return;
  }
  CHOMP(command);
  c = (char)tolower((int)command[0]);
  if (c == 'w') {
    color = S_WHITE;
  } else if (c == 'b') {
    color = S_BLACK;
  } else {
    GTP_response(err_genmove, true);
    return;
  }

  player_color = color;

  if (sim_move)
    point = SimulationGenmove(game, color);
  else
    point = UctSearchGenmove(game, color);

  stringstream out;
  PrintBestSequenceGFX(out, game, uct_node, current_root, color);
  GTP_response(out.str().c_str(), true);
}

////////////////////////////
//  void GTP_ray_param()  //
////////////////////////////
static void
GTP_ray_param()
{
  char *command;

  command = STRTOK(NULL, DELIM, &next_token);
  if (command == NULL) {
    GTP_response("ray-param name value", false);
    return;
  }
  CHOMP(command);
  string name(command);

  command = STRTOK(NULL, DELIM, &next_token);
  if (command == NULL) {
    GTP_response("ray-param name value", false);
    return;
  }
  CHOMP(command);
  double value = atof(command);

  if (name == "c_puct") {
    c_puct = value;
  } else if (name == "value_scale") {
    value_scale = value;
  } else if (name == "expand_threshold") {
    custom_expand_threshold = value;
  } else if (name == "policy_temperature") {
    policy_temperature = value;
  } else if (name == "tgr1_rate") {
    tgr1_rate = round(value);
  } else if (name == "lgrf1_rate") {
    lgrf1_rate = round(value);
  } else if (name == "resign_threshold") {
    resign_threshold = value;
  } else {
    GTP_response("unknown param", false);
    return;
  }

  GTP_response(brank, true);
}

///////////////////////////
//  void GTP_ray_stat()  //
///////////////////////////
static void
GTP_ray_stat()
{
  char *command;
  char c;
  char pos[10];
  int color;
  int point = PASS;

  command = STRTOK(input_copy, DELIM, &next_token);

  CHOMP(command);
  command = STRTOK(NULL, DELIM, &next_token);
  if (command == NULL) {
    GTP_response(err_genmove, true);
    return;
  }
  CHOMP(command);
  c = (char)tolower((int)command[0]);
  if (c == 'w') {
    color = S_WHITE;
  } else if (c == 'b') {
    color = S_BLACK;
  } else {
    GTP_response(err_genmove, true);
    return;
  }

  player_color = color;

  if (sim_move)
    point = SimulationGenmove(game, color);
  else
    point = UctSearchGenmove(game, color);

  stringstream out;
  PrintMoveStat(out, game, uct_node, current_root);

  GTP_response(out.str().c_str(), true);
  //cout << endl << endl;
}
 
static int features_turn_count = 0;
static int features_turn_next = 1;
void DumpFeature(const uct_node_t& node, int color, int move, int win);

void GTP_features_planes_file(void)
{
  char *command;

  command = STRTOK(NULL, DELIM, &next_token);
  if (command == NULL || game->moves == 0) {
    GTP_response(err_genmove, true);
    return;
  }

  int color = game->record[game->moves - 1].color;
  int move = game->record[game->moves - 1].pos;

  CHOMP(command);
  char c = (char)tolower((int)command[0]);
  int win;
  if (c == 'w') {
    win = color == S_WHITE ? 1 : -1;
#if 0
    if (store_winning_percentage > 0.40) {
      cerr << "####### SKIP " << c << " " << store_winning_percentage << endl;
      //GTP_response(brank, true);
      //return;
    } else {
      cerr << "####### DUMP " << c << " " << store_winning_percentage << endl;
    }
#endif
  } else if (c == 'b') {
    win = color == S_BLACK ? 1 : -1;
#if 0
    if (store_winning_percentage < 0.60) {
      cerr << "####### SKIP " << c << " " << store_winning_percentage << endl;
      //GTP_response(brank, true);
      //return;
    } else {
      cerr << "####### DUMP " << c << " " << store_winning_percentage << endl;
    }
#endif
  } else {
    GTP_response(err_genmove, true);
    return;
  }

  if (move == RESIGN || move == PASS) {
    GTP_response(brank, true);
    return;
  }

  DumpFeature(store_node, color, move, win);

  GTP_response(brank, true);
}

static void DumpSparse(std::ostream& stream, const std::vector<float> data)
{
  for (auto i = 0; i < data.size(); i++) {
    float f = data[i];
    if (i == 0 || f != 0) {
      stream << i << ':' << f << ' ';
    }
  }
}

static void DumpDense(std::ostream& stream, const std::vector<float> data)
{
  for (auto i = 0; i < data.size(); i++) {
    stream << data[i] << ' ';
  }
}

void DumpFeature(const uct_node_t& node, int color, int move, int win)
{
  double rate[PURE_BOARD_MAX];
  AnalyzePoRating(game_prev, color, rate);

  std::vector<float> data_basic;
  std::vector<float> data_features;
  std::vector<float> data_history;
  std::vector<float> data_owner;

  int t = rand() / 11 % 8;
  //static int t = 0; t++;
  int moveT = RevTransformMove(move, t);
  WritePlanes(data_basic, data_features, data_history, &data_owner,
    game_prev, &node, color, t);

  int x = CORRECT_X(moveT) - 1;
  int y = CORRECT_Y(moveT) - 1;
  int label = x + y * pure_board_size;
  if (label < 0 || label > 19 * 19) {
    cerr << "bad label " << x << " " << y << endl;
    abort();
  }
  if (isnan(data_owner[0])) {
    cerr << "bad stat" << endl;
    return;
  }

  if (!stream_ptr) {
    stream_ptr = make_unique<ofstream>("data.txt", std::ios::app | std::ios::binary);
  }
  auto& stream = *stream_ptr;

  stream << "|win " << win;
  stream << "|move " << label << ":1";
  stream << "|color " << color - 1;
  stream << "|komi " << komi[0];
  stream << "|basic ";
  DumpSparse(stream, data_basic);
  stream << "|features ";
  DumpSparse(stream, data_features);
  stream << "|history ";
  DumpSparse(stream, data_history);
#if 1
  stream << "|statistic ";
  DumpDense(stream, data_owner);
#endif
  stream << endl;
}

void GTP_features_clear(void)
{
  GTP_response(brank, true);
  return;
}

void GTP_features_store(void)
{
  char *command;

  int color = FLIP_COLOR(game->record[game->moves - 1].color);

  CopyGame(store_game, game);
  UctSearchStat(store_game, color, 100);

  const uct_node_t *root = &uct_node[current_root];
  memcpy(&store_node, root, sizeof(uct_node_t));
  double winning_percentage = (double)root->win / root->move_count;
  if (color == S_BLACK) {
    store_winning_percentage = winning_percentage;
  } else {
    store_winning_percentage = 1 - winning_percentage;
  }

  GTP_response(brank, true);
}


///////////////////////
//  void GTP_stat()  //
///////////////////////
void
GTP_stat(void)
{
  char *command;
  char c;
  char pos[10];
  int color;
  int point = PASS;

  command = STRTOK(input_copy, DELIM, &next_token);

  CHOMP(command);
  command = STRTOK(NULL, DELIM, &next_token);
  if (command == NULL) {
    GTP_response(err_genmove, true);
    return;
  }
  CHOMP(command);
  c = (char)tolower((int)command[0]);
  if (c == 'w') {
    color = S_WHITE;
  } else if (c == 'b') {
    color = S_BLACK;
  } else {
    GTP_response(err_genmove, true);
    return;
  }

  player_color = FLIP_COLOR(game->record[game->moves - 1].color);

  int win = player_color == color ? 1 : 0;

  //player_color = color;

  cerr << "VALUE";
  double value_sum = 0;
  for (int i = 0; i < 8; i++) {
    std::vector<float> data, data2;
    int t = i;// rand() / 11 % 8;
    std::vector<float> data_basic;
    std::vector<float> data_features;
    std::vector<float> data_history;
    std::vector<float> data_owner;

    WritePlanes(data_basic, data_features, data_history, &data_owner,
      game, &store_node, player_color, t);

    std::vector<int> eval_node_index;
    std::vector<int> eval_node_color;
    std::vector<int> eval_node_trans;
    std::vector<int> eval_node_path;
    //std::vector<float> eval_input_data;

    eval_node_index.push_back(current_root);
    eval_node_color.push_back(player_color);
    eval_node_trans.push_back(t);
    //std::copy(req.data.begin(), req.data.end(), std::back_inserter(eval_input_data));
    //std::copy(req.path.rbegin(), req.path.rend(), std::back_inserter(eval_node_path));
    eval_node_path.push_back(-1);
#if 0
    //TODO
    uct_node[current_root].value = -1;
    EvalUctNode(eval_node_index, eval_node_color, eval_node_trans, data, eval_node_path);
    double value = uct_node[current_root].value;
    double se_value = abs(value - win);
    cerr << '\t' << se_value;
    value_sum += uct_node[current_root].value;
    uct_node[current_root].value = -1;
    //<< '\t' << data[19 * 3 + 3] << ':' << POS(10, 11) << ':' << moveT;
#endif
  }
  cerr << endl;

  point = UctSearchGenmove(game, player_color);
  if (point != RESIGN) {
    //PutStone(game, point, color);
  }


  uct_node_t *root = &uct_node[current_root];
  double winning_percentage = (double)root->win / root->move_count;
  //double value = root->value;
  double valuet = (double)root->value_win / root->value_move_count;
  double se_po = abs(winning_percentage - win);
  //double se_value = abs(value - win);
  double se_valuet = abs(valuet - win);
  double se_value8 = abs(value_sum / 8 - win);
  cerr << "STAT\t" << game->moves << "\t" << winning_percentage
    //<< "\t" << value
    << "\t" << se_po
    //<< "\t" << se_value
    << '\t' << se_value8
    << '\t' << root->value_move_count
    << '\t' << se_valuet
    << endl;

  IntegerToString(point, pos);

  GTP_response(pos, true);

  //UctSearchPondering(game, FLIP_COLOR(color));
}



////////////////////////////////
//  シミュレーションの検証        //
////////////////////////////////
void
GTP_stat_po(void)
{
  char *command;
  //char c;
  //char pos[10];
  //int color;
  //int point = PASS;

  command = STRTOK(input_copy, DELIM, &next_token);

  CHOMP(command);
  command = STRTOK(NULL, DELIM, &next_token);
  if (command == NULL) {
    GTP_response(err_genmove, true);
    return;
  }
  /*
  CHOMP(command);
  c = (char)tolower((int)command[0]);
  if (c == 'w') {
    color = S_WHITE;
  } else if (c == 'b') {
    color = S_BLACK;
  } else {
    GTP_response(err_genmove, true);
    return;
  }
  */

  int color = game->record[game->moves - 1].color;
  int move = game->record[game->moves - 1].pos;
  //void
  //Simulation(game_info_t *game, int starting_color, std::mt19937_64 *mt)
  //{
    //int color = starting_color;
  int pos = -1;
  int length;
  int pass_count;

  // レートの初期化  
  game_prev->sum_rate[0] = game_prev->sum_rate[1] = 0;
  memset(game_prev->sum_rate_row, 0, sizeof(long long) * 2 * BOARD_SIZE);
  memset(game_prev->rate, 0, sizeof(long long) * 2 * BOARD_MAX);

  pass_count = (game_prev->record[game_prev->moves - 1].pos == PASS && game_prev->moves > 1);


  auto begin_time = ray_clock::now();
  for (int i = 0; i < 1000; i++) {
    // 黒番のレートの計算
    Rating(game_prev, S_BLACK, &game_prev->sum_rate[0], game_prev->sum_rate_row[0], game_prev->rate[0]);
    // 白番のレートの計算
    Rating(game_prev, S_WHITE, &game_prev->sum_rate[1], game_prev->sum_rate_row[1], game_prev->rate[1]);
  }
  auto finish_time = GetSpendTime(begin_time) * 1000;

  long long *rate = game_prev->rate[color - 1];

  long long max_rate = 0;
  int max_pos = PASS;
  for (int i = 0; i < pure_board_max; i++) {
    int pos = onboard_pos[i];

    if (IsLegalNotEye(game_prev, pos, color)) {
      long long r = rate[pos];
      if (r > max_rate) {
	max_rate = r;
	max_pos = pos;
      }
    }
  }

  if (max_pos == move) {
    cerr << "####### PO HIT " << finish_time << endl;
  } else {
    cerr << "####### PO MISS " << finish_time << endl;
  }
#if 0
  // 終局まで対局をシミュレート
  while (length-- && pass_count < 2) {
    // 着手を生成する
    pos = RatingMove(game, color, mt);
    // 石を置く
    PoPutStone(game, pos, color);
    // パスの確認
    pass_count = (pos == PASS) ? (pass_count + 1) : 0;
    // 手番の入れ替え
    color = FLIP_COLOR(color);
  }
#endif
}
