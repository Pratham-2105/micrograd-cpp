#include "nn.hpp"
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

  /*
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

  auto x = std::make_shared<Value>(3.0);
  auto y = add(x, x);

  y->backward();

  std::cout << "x->grad = " << x->grad << "  (expect 2)\n";

  */

  /*
  auto z = std::make_shared<Value>(0.0);
  auto t = tanh_(z);

  std::cout << "t->data = " << t->data << "  (expect 0)\n";
  t->backward();
  std::cout << "z->grad = " << z->grad << "  (expect 1)\n";

  */

  /*
  Neuron n(3);
  std::cout << "w.size() = " << n.w.size() << "  (expect 3)\n";
  std::cout << "b->data  = " << n.b->data << "  (in [-1,1])\n";
  std::cout << "w[0]->data = " << n.w[0]->data << "  (in [-1,1])\n";
 */

  Neuron n(2);
  std::vector<std::shared_ptr<Value>> x = {std::make_shared<Value>(1.0),
                                           std::make_shared<Value>(-1.0)};

  auto out = n.forward(x);
  std::cout << "out->data = " << out->data << "  (in (-1,1))\n";

  out->backward();

  std::cout << "w[0]->grad = " << n.w[0]->grad << "  (nonzero)\n";
  std::cout << "b->grad    = " << n.b->grad << "  (expect 1... ish)\n";

  return 0;
}
