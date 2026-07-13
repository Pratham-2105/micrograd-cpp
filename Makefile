CXX      := g++
CXXFLAGS := -std=c++17 -Wall -Wextra -Wpedantic -g
TARGET   := micrograd
SRCS     := main.cpp
HEADERS  := value.hpp nn.hpp

$(TARGET): $(SRCS) $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRCS)

run: $(TARGET)
	./$(TARGET)

test: test.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -o test test.cpp
	./test

clean:
	rm -f $(TARGET) test

.PHONY: run test clean
