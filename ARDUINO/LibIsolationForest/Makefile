CFLAGS=-Wall -O3 -std=c++11

.PHONY: all clean 

all : IsolationForestiForestpy

IsolationForestiForestpy : main-iForestpy.o IsolationForestiForestpy.o 
	${CXX} ${CFLAGS} $^ -o $@ ${LDFLAGS} 

main-iForestpy.o : main-iForestpy.cpp
	${CXX} ${CFLAGS} -c $^ -o $@

IsolationForestiForestpy.o : IsolationForestiForestpy.cpp
	${CXX} ${CFLAGS} -c $^ -o $@

clean : 
	#-rm -f *.o IsolationForestiForestpy
	-rm -f *.o 

