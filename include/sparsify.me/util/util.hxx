/**
 * @file util.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief
 * @version 0.1
 * @date 2021-05-17
 *
 * @copyright Copyright (c) 2021
 *
 */
#pragma once
#include <random>

namespace sparsifyme {
namespace util {

template <typename type_t = float>
type_t get_random(type_t begin = 0.0f, type_t end = 1.0f) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(begin, end);
  return (type_t)dis(gen);
}

template <typename type_t>
type_t ceil_div(type_t x, type_t y) {
  return (x + y - 1) / y;
}

}  // namespace util
}  // namespace sparsifyme

#include <sparsify.me/util/timer.hxx>
#include <sparsify.me/util/launch.hxx>
