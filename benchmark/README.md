# Benchmark
This program compares the throughputs of the generators implemented in the file "src/include/generators.hpp".

Each generator is tested three times: when generating numbers of type *uint32_t*, when generating numbers of type *float* in \[0, 1), and when generating numbers of type *double* in \[0, 1).

# Compilation

To compile the benchmarks, run the makefile with the argument `ARCH=xx` where `xx` must correspond to the version of the architecture, or "compute capabiliy", of your CUDA-enabled GPU. For example, if the device has compute capability 7.0,

    make ARCH=70

Then, to execute the resulting binaries, run the following commands where `<x>` must be replaced with the value of the parameter *x*.

    ./bin/SumMRGk5_93 <numRepetitions> <threadsPerBlock> <numBlocks> <deviceId> <printHeader>
    ./bin/SumCombinedMRGs <numRepetitions> <threadsPerBlock> <numBlocks> <deviceId> <printHeader>
    ./bin/SumOtherMRGs <numRepetitions> <threadsPerBlock> <numBlocks> <deviceId> <printHeader>
    
    ./bin/SumMWCs <numRepetitions> <threadsPerBlock> <numBlocks> <deviceId> <printHeader>
    
    ./bin/SumLFSRs <numRepetitions> <threadsPerBlock> <numBlocks> <deviceId> <printHeader>
    ./bin/SumOtherF2 <numRepetitions> <threadsPerBlock> <numBlocks> <deviceId> <printHeader>
    
    ./bin/SumCBRNGs <numRepetitions> <threadsPerBlock> <numBlocks> <deviceId> <printHeader>
