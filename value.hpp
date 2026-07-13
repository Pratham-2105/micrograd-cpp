#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <set>
#include <vector>

typedef int8_t i8;
typedef int16_t i16;
typedef int32_t i32;
typedef int64_t i64;

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

typedef float f32;
typedef double f64;

using Scalar = f64;

struct Value {
  Scalar data, grad = 0;
  std::vector<std::shared_ptr<Value>> children;
  std::function<void()> _backward = []() {};

  Value(Scalar num) { data = num; }

  Value(Scalar num, std::vector<std::shared_ptr<Value>> child)
      : data(num), children(child) {}

  void build_topo(std::vector<Value *> &topo, std::set<Value *> &visited) {
    if (visited.count(this))
      return;
    visited.insert(this);

    for (auto &child : children) {
      child->build_topo(topo, visited);
    }

    topo.push_back(this);
  }

  void backward() {
    std::vector<Value *> topo;
    std::set<Value *> visited;
    build_topo(topo, visited);

    this->grad = 1.0;
    for (i64 it = topo.size() - 1; it >= 0; --it) {
      auto child = topo[it];
      child->_backward();
    }
  }
};

std::shared_ptr<Value> add(std::shared_ptr<Value> a, std::shared_ptr<Value> b) {
  auto out = std::make_shared<Value>(a->data + b->data,
                                     std::vector<std::shared_ptr<Value>>{a, b});

  Value *out_raw = out.get();
  out->_backward = [a, b, out_raw]() {
    a->grad += 1 * out_raw->grad;
    b->grad += 1 * out_raw->grad;
  };

  return out;
}

std::shared_ptr<Value> mul(std::shared_ptr<Value> a, std::shared_ptr<Value> b) {

  auto out = std::make_shared<Value>(a->data * b->data,
                                     std::vector<std::shared_ptr<Value>>{a, b});
  Value *out_raw = out.get();
  out->_backward = [a, b, out_raw]() {
    a->grad += (b->data) * out_raw->grad;
    b->grad += (a->data) * out_raw->grad;
  };

  return out;
}
