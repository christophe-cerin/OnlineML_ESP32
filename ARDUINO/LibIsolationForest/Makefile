CFLAGS=-Wall -O3 -std=c++11

.PHONY: all clean 

all : IsolationForest

IsolationForest : main.o IsolationForest.o 
	${CXX} ${CFLAGS} $^ -o $@ ${LDFLAGS} 

main.o : main.cpp
	${CXX} ${CFLAGS} -c $^ -o $@

IsolationForest.o : IsolationForest.cpp
	${CXX} ${CFLAGS} -c $^ -o $@

clean : 
	#-rm -f *.o IsolationForest
	-rm -f *.o 
