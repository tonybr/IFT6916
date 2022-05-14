# cuRNG v. 1.0
*cuRNG* is a header-only library written in C++ that provides classes for the creation of streams, and substreams, of uniform random numbers on CUDA-enabled GPUs. It also offers pseudo-random number generators that were meticulously optimized for this platform.

**Important:** the pseudo-random number generators in *cuRNG* should not be used for now except for *Xoshiro128*++. In fact, the generators *MRG31* and *MWC32* are the results of a research project at Université de Montréal in which traditional implementations of recurrence-based generators were optimized for CUDA-enabled GPUs by performing extensive performance tests. However, the parameters of both generators are arbitrary, and thorough analyses, which were not part of the project, will be required to find adequate parameters that yield fully adequately distributed vectors of random numbers.

# Compilation

To compile the examples included in *cuRNG* as well as the tests, run the makefile with the argument `ARCH=xx` where `xx` must correspond to the version of the architecture, or "compute capabiliy", of your CUDA-enabled GPU. For example, if the device has compute capability 7.0,

    make ARCH=70

Then, to execute the resulting binaries, run the following commands where `<x>` must be replaced with the value of the parameter *x*.

    ./bin/runTests
    ./bin/createStreams <numRepetitions> <numStreams>
    ./bin/generateUniform01 <numRepetitions> <numVariates>
    ./bin/simulateInventory <numRepetitions> <numSimulations> <numDays>

Finally, to generate the documentation for *cuRNG*, run the makefile with the target `docs`.
