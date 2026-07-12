#pragma once

#include <cstdint>

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

  Value(Scalar num) { data = num; }

  Value operator+(const Value &other) { return Value(data + other.data); }

  Value operator*(const Value &other) { return Value(data * other.data); }
};
