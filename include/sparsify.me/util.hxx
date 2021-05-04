#include <random>

#pragma once
namespace sparsifyme {
namespace util {

float get_random() {
  static std::default_random_engine e;
  static std::uniform_real_distribution<> dis(0, 1);  // rage 0 - 1
  return dis(e);
}

}  // namespace util
}  // namespace sparsifyme