CXX = g++
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra

TARGET = main
SRC = $(wildcard *.cpp)
OBJ = $(patsubst %.cpp, %.o, $(SRC))

$(TARGET): $(OBJ)
	$(CXX) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -fno-tree-vectorize -c $< -o $@

.PHONY: clean
clean:
	rm -f *.o $(TARGET)
