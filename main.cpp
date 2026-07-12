#include "value.hpp"
#include <iostream>

int main() {
  Value a(2), b(3);
  Value c = a + b;
  Value d = a * b;

  std::cout << "c: " << c.data << " d: " << d.data << "\n";

  return 0;
}
