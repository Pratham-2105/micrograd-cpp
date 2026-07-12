#pragma once

#include <cstdint>
#include <memory>
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
  Scalar data;
  std::vector<std::shared_ptr<Value>> children;

  Value(Scalar num) { data = num; }

  Value(Scalar num, std::vector<std::shared_ptr<Value>> child)
      : data(num), children(child) {}

  Value operator+(const Value &other) { return Value(data + other.data); }

  Value operator*(const Value &other) { return Value(data * other.data); }
};

std::shared_ptr<Value> mul(std::shared_ptr<Value> a, std::shared_ptr<Value> b) {

  return std::make_shared<Value>(a->data * b->data,
                                 std::vector<std::shared_ptr<Value>>{a, b});
}
