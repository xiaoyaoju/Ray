#pragma once

#ifdef LEELA

struct record_t {
  int color;
  int pos;
  unsigned long long hash;
};

#else

struct Netresult {
  // 19x19 board positions
  std::vector<float> policy;

  // pass
  float policy_pass;

  // winrate
  float winrate;

  Netresult();
};

#endif

int InitializeLeela();

Netresult EvaluateLeela(int moves, record_t *record);
