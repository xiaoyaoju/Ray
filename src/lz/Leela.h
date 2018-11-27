#pragma once

#include <array>

#ifdef LEELA

struct record_t {
  int color;
  int pos;
  unsigned long long hash;
};

#else

#endif


#define NUM_INTERSECTIONS (19 * 19)

struct Netresult {
  // 19x19 board positions
  std::array<float, NUM_INTERSECTIONS> policy;

  // pass
  float policy_pass;

  // winrate
  float winrate;

  Netresult() : policy_pass(0.0f), winrate(0.0f) {
    policy.fill(0.0f);
  }
};

int InitializeLeela();

Netresult EvaluateLeela(int moves, record_t *record);
