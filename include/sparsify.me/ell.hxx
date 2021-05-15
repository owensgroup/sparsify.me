#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <sparsify.me/util.hxx>

#pragma once

namespace sparsifyme {
using namespace util;
template <typename type_t = float,
          memory_space_t space = memory_space_t::device>
struct ell_t {
  std::size_t rows, cols, block_size;  /// m, k and user-defined block-size
  std::size_t ell_cols;                /// ...
  std::size_t blocked_rows;            /// rows / block_size
  std::size_t blocked_cols;            /// ell_cols / block_size
  std::size_t num_blocks;              /// blocked_rows * blocked_cols

  // Storage for column indices and ell values (blocked)
  vector_t<std::size_t, space>
      column_indices;              /// [blocked_rows x blocked_cols]
  vector_t<type_t, space> values;  /// [rows x ell_cols]

  ell_t() {}
  ~ell_t() {}

  template <memory_space_t in_space>
  ell_t<type_t, space>& operator=(const ell_t<type_t, in_space>& rhs) {
    rows = rhs.rows;
    cols = rhs.cols;
    block_size = rhs.block_size;
    ell_cols = rhs.ell_cols;
    blocked_rows = rhs.blocked_rows;
    blocked_cols = rhs.blocked_cols;
    num_blocks = rhs.num_blocks;
    column_indices = rhs.column_indices;
    values = rhs.values;
    return *this;
  }

  void print() {
    std::cout << "A-Matrix" << std::endl;
    std::cout << "\t(rows, cols) = " << rows << ", " << cols << std::endl;
    std::cout << "\tELL columns = " << ell_cols << std::endl;
    std::cout << "\tBlock Size = " << block_size << std::endl;
    std::cout << "\tNumber of Blocks = " << num_blocks << std::endl;
    std::cout << "\tColumn Idx = ";
    thrust::copy(column_indices.begin(), column_indices.end(),
                 std::ostream_iterator<std::size_t>(std::cout, " "));
    std::cout << std::endl;
    std::cout << "\tValues = ";
    thrust::host_vector<type_t> temp = values;
    for(auto&val : temp)
      std::cout << static_cast<float>(val) << " ";
    std::cout << std::endl;
  }
};
}  // namespace sparsifyme