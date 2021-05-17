/**
 * @file vector.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief
 * @version 0.1
 * @date 2021-05-17
 *
 * @copyright Copyright (c) 2021
 *
 */
#pragma once
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <sparsify.me/containers/memory.hxx>

namespace sparsifyme {
template <typename type_t, memory_space_t space>
using vector_t =
    std::conditional_t<space == memory_space_t::host,  // condition
                       thrust::host_vector<type_t>,    // host_type
                       thrust::device_vector<type_t>   // device_type
                       >;
}