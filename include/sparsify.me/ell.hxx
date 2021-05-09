#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <sparsify.me/util.hxx>

#pragma once

namespace sparsifyme {
using namespace util;
template <typename type_t = float,
          memory_space_t space = memory_space_t::device>
struct ell_t {
  std::size_t rows, cols, block_size, num_blocks;
  vector_t<std::size_t, space> column_indices;
  vector_t<type_t, space> values;

  template <memory_space_t in_space>
  ell_t<type_t, space>& operator=(const ell_t<type_t, in_space>& rhs) {
    rows = rhs.rows;
    cols = rhs.cols;
    block_size = rhs.block_size;
    num_blocks = rhs.num_blocks;
    column_indices = rhs.column_indices;
    values = rhs.values;
    return *this;
  }
};
}  // namespace sparsifyme