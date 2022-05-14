
HEADERS = $(wildcard ./include/cuRNG/*)
ARGUMENTS = -arch=compute_$(ARCH) -code=sm_$(ARCH)

binaries: ./bin/runTests ./bin/createStreams ./bin/simulateInventory ./bin/generateUniform01

docs: ./docs/html/

clean:
	rm -f ./bin/runTests
	rm -f ./bin/createStreams
	rm -f ./bin/simulateInventory
	rm -f ./bin/generateUniform01

./bin/runTests: ./tests/runTests.cpp $(HEADERS)
	mkdir -p bin
	nvcc $(ARGUMENTS) -x cu $< -o $@

./bin/createStreams: ./examples/createStreams.cpp $(HEADERS)
	mkdir -p bin
	nvcc $(ARGUMENTS) -x cu $< -o $@

./bin/simulateInventory: ./examples/simulateInventory.cpp $(HEADERS)
	mkdir -p bin
	nvcc $(ARGUMENTS) -x cu $< -o $@

./bin/generateUniform01: ./examples/generateUniform01.cpp $(HEADERS)
	mkdir -p bin
	nvcc $(ARGUMENTS) -lcurand -x cu $< -o $@

./docs/html/: $(HEADERS)
	cd ./docs; doxygen
