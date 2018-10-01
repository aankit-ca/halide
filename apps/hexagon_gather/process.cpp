#include <stdio.h>
#include <memory.h>
#include <assert.h>
#include <stdlib.h>
#include "pipeline.h"
#include "HalideBuffer.h"
#include "process.h"

#ifdef HL_HEXAGON_DEVICE
#include "halide_benchmark.h"
#include "HalideRuntimeHexagonHost.h"
#include "halide_image_io.h"
using namespace Halide::Runtime;
using namespace Halide::Tools;
#else
#include "simulator_benchmark.h"
#include "io.h"
#endif

// Verify result for the halide pipeline
int checker(Halide::Runtime::Buffer<DTYPE> &in,
        Halide::Runtime::Buffer<DTYPE> &lut,
        Halide::Runtime::Buffer<DTYPE> &out) {
    int errcnt = 0, maxerr = 10;
    printf("Checking...\n");

    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            DTYPE xidx = std::min(std::max((int)in(x, 0), 0), W-1);
            DTYPE yidx = std::min(std::max((int)in(x, 1), 0), H-1);
            if (out(x, y) != lut(xidx, yidx)) {
                errcnt++;
                // if (errcnt <= maxerr) {
                    printf("Mismatch at (%4d, %4d): %3d (Halide) == %3d (Expected)\n",
                            x, y, out(x, y), lut(xidx, yidx));
                // }
            }
        }
    }
    if (errcnt > maxerr) {
        printf("...\n");
    }
    if (errcnt > 0) {
        printf("Mismatch at %d places\n", errcnt);
    }
    return errcnt;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: %s (iterations)\n", argv[0]);
        return 0;
    }
    int iterations = atoi(argv[1]);

#ifdef HL_HEXAGON_DEVICE
    Halide::Runtime::Buffer<DTYPE>  in(nullptr, W, H);
    Halide::Runtime::Buffer<DTYPE> out(nullptr, W, H);
    Halide::Runtime::Buffer<DTYPE> lut(nullptr, W, H);

    in.device_malloc(halide_hexagon_device_interface());
    out.device_malloc(halide_hexagon_device_interface());
    lut.device_malloc(halide_hexagon_device_interface());
#else
    DTYPE *in_ptr  = (DTYPE *)memalign(1 << LOG2VLEN, W*H*sizeof(DTYPE));
    DTYPE *out_ptr = (DTYPE *)memalign(1 << LOG2VLEN, W*H*sizeof(DTYPE));
    DTYPE *lut_ptr = (DTYPE *)memalign(1 << LOG2VLEN, w*H*sizeof(DTYPE));

    Halide::Runtime::Buffer<DTYPE> in(in_ptr, W, H);
    Halide::Runtime::Buffer<DTYPE> out(out_ptr, W, H);
    Halide::Runtime::Buffer<DTYPE> lut(lut_ptr, W, H);
#endif

    srand(0);
    // Fill the input image
    in.for_each_value([&](DTYPE &x) {
        x = static_cast<DTYPE>(rand());
    });
    // Fill the lookup table
    lut.for_each_value([&](DTYPE &x) {
        x = static_cast<DTYPE>(rand());
    });

#ifdef HL_HEXAGON_DEVICE
    // To avoid the cost of powering HVX on in each call of the
    // pipeline, power it on once now. Also, set Hexagon performance to turbo.
    halide_hexagon_set_performance_mode(NULL, halide_hexagon_power_turbo);
    halide_hexagon_power_hvx_on(NULL);
#endif

    printf("Running pipeline...\n\n");
    printf("Image size: %d pixels\n", W*H);
    printf("Image type: %d bits\n", (int) sizeof(DTYPE)*8);
    printf("Table size: %d elements\n\n", W*H);

#ifdef HL_HEXAGON_DEVICE
    double time = Halide::Tools::benchmark(iterations, 1, [&]() {
#else
    double time = benchmark([&]() {
#endif
    int result = pipeline(in, lut, out);
        if (result != 0) {
            printf("pipeline failed! %d\n", result);
        }
    });
    out.copy_to_host();
    printf("Done, TIME: %g ms\nTHROUGHPUT: %g MP/s\n", time*1000.0, ((double)(W*H))/((double)(1000000) * time));

#ifdef HL_HEXAGON_DEVICE
    // We're done with HVX, power it off, and reset the performance mode
    // to default to save power.
    halide_hexagon_power_hvx_off(NULL);
    halide_hexagon_set_performance_mode(NULL, halide_hexagon_power_default);
#endif

#if 1
    if (checker(in, lut, out) != 0) {
        printf("Fail!\n");
        return 1;
    }
#endif
    printf("Success!\n");

    return 0;
}
