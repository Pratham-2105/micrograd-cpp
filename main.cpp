#include "value.hpp"
#include <iostream>
#include <memory>

int main() {
  /*
    Value a(2), b(3);
    Value c = a + b;
    Value d = a * b;

    std::cout << "c: " << c.data << " d: " << d.data << "\n";
  */

  /*
  auto a = std::make_shared<Value>(2.0);
  auto b = std::make_shared<Value>(3.0);

  auto c = mul(a, b);

  std::cout << "c->data            = " << c->data << "  (expect 6)\n";
  std::cout << "c->children[0]->data = " << c->children[0]->data
            << "  (expect 2)\n";
  std::cout << "c->children[1]->data = " << c->children[1]->data
            << "  (expect 3)\n";

  bool same = (c->children[0].get() == a.get());
  std::cout << "child0 is same object as a? " << (same ? "true" : "false")
            << "  (expect true)\n";

  */
  /*

  auto a = std::make_shared<Value>(2.0);
  auto b = std::make_shared<Value>(3.0);
  auto c = mul(a, b);

  c->grad = 1.0;
  c->_backward();

  std::cout << "a->grad = " << a->grad << "  (expect 3)\n";
  std::cout << "b->grad = " << b->grad << "  (expect 2)\n";

  auto x = std::make_shared<Value>(2.0);
  auto y = std::make_shared<Value>(3.0);
  auto s = add(x, y);

  s->grad = 1.0;
  s->_backward();

  std::cout << "x->grad = " << x->grad << "  (expect 1)\n";
  std::cout << "y->grad = " << y->grad << "  (expect 1)\n";
  */

  auto a = std::make_shared<Value>(2.0);
  auto b = std::make_shared<Value>(3.0);
  auto c = std::make_shared<Value>(10.0);
  auto e = mul(a, b); // 6
  auto L = add(e, c); // 16

  L->backward();

  std::cout << "L->data = " << L->data << "  (expect 16)\n";
  std::cout << "c->grad = " << c->grad << "  (expect 1)\n";
  std::cout << "a->grad = " << a->grad << "  (expect 3)\n";
  std::cout << "b->grad = " << b->grad << "  (expect 2)\n";

  return 0;
}
