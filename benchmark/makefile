
HEADERS = $(wildcard ./src/include/*)
ARGUMENTS = -arch=compute_$(ARCH) -code=sm_$(ARCH)

FILES_CPP = $(wildcard ./src/*.cpp)
FILES_EXE = $(patsubst ./src/%.cpp, ./bin/%, $(FILES_CPP))
FILES_PTX = $(patsubst ./src/%.cpp, ./src/%.ptx, $(FILES_CPP))

binaries: $(FILES_EXE)

generatePTX: binaries $(FILES_PTX)

clean:
	rm -f $(FILES_EXE)
	rm -f $(FILES_PTX)

./bin/%: ./src/%.cpp $(HEADERS)
	mkdir -p bin
	nvcc $(ARGUMENTS) -x cu $< -o $@

./src/%.ptx: ./src/%.cpp $(HEADERS)
	nvcc $(ARGUMENTS) -DUNROLL_1 -x cu -ptx $< -o $@
