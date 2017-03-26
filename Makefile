FLAGS = -g -O3 -mavx -march=native -Wall -Wextra -std=c++11
OBJS = obj/main.o obj/neuralnet.o obj/layer.o obj/helper.o
HEADER = -Iinclude
PROG = neural
CXX = g++

all: $(PROG)

obj/%.o: src/%.cpp
	@mkdir -p obj
	$(CXX) $(CFLAGS) -c -s $< $(HEADER) $(FLAGS)
	@mv *.o obj/

$(PROG): $(OBJS)
	$(CXX) $(OBJS) -o $(PROG) $(FLAGS) 

clean:
	rm -f $(PROG)
	rm -rf obj/
