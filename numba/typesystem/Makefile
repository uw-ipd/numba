CXX = clang++
CXXFLAGS = -Wall

all: test

test: typesystem.hpp typesystem.cpp test.cpp MurmurHash2.cpp
	$(CXX) $(CXXFLAGS) -o $@ typesystem.cpp test.cpp MurmurHash2.cpp