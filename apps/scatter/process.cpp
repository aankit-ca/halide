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

// DTYPE clamp(DTYPE a, DTYPE min_val, DTYPE max_val) {
//     return std::max(std::min(a, max_val), min_val);
// }

void swap(Halide::Runtime::Buffer<DTYPE> &lut, int idx1, int idx2) {
    DTYPE tmp = lut(idx1);
    lut(idx1) = lut(idx2);
    lut(idx2) = tmp;
}

// Verify result for the halide pipeline
int checker(Halide::Runtime::Buffer<DTYPE> &input,
            Halide::Runtime::Buffer<DTYPE> &lut1,
            Halide::Runtime::Buffer<DTYPE> &lut2,
            Halide::Runtime::Buffer<DTYPE> &output) {
    int errcnt = 0, maxerr = 10;
    printf("Checking...\n");

#ifdef HL_HEXAGON_DEVICE
    Halide::Runtime::Buffer<DTYPE> buf0(nullptr, W);
    Halide::Runtime::Buffer<DTYPE> buf1(nullptr, W);
    Halide::Runtime::Buffer<DTYPE> buf2(nullptr, W);
    Halide::Runtime::Buffer<DTYPE> buf3(nullptr, W);
    Halide::Runtime::Buffer<DTYPE> buf4(nullptr, W);
    Halide::Runtime::Buffer<DTYPE> buf5(nullptr, W);

    buf0.device_malloc(halide_hexagon_device_interface());
    buf2.device_malloc(halide_hexagon_device_interface());
    buf1.device_malloc(halide_hexagon_device_interface());
    buf3.device_malloc(halide_hexagon_device_interface());
    buf4.device_malloc(halide_hexagon_device_interface());
    buf5.device_malloc(halide_hexagon_device_interface());
#else
    DTYPE *buf0_ptr = (DTYPE *) memalign(1 << LOG2VLEN, W*sizeof(DTYPE));
    DTYPE *buf1_ptr = (DTYPE *) memalign(1 << LOG2VLEN, W*sizeof(DTYPE));
    DTYPE *buf2_ptr = (DTYPE *) memalign(1 << LOG2VLEN, W*sizeof(DTYPE));
    DTYPE *buf3_ptr = (DTYPE *) memalign(1 << LOG2VLEN, W*sizeof(DTYPE));
    DTYPE *buf4_ptr = (DTYPE *) memalign(1 << LOG2VLEN, W*sizeof(DTYPE));
    DTYPE *buf5_ptr = (DTYPE *) memalign(1 << LOG2VLEN, W*sizeof(DTYPE));
    Halide::Runtime::Buffer<DTYPE> buf0(buf0_ptr, W);
    Halide::Runtime::Buffer<DTYPE> buf1(buf1_ptr, W);
    Halide::Runtime::Buffer<DTYPE> buf2(buf2_ptr, W);
    Halide::Runtime::Buffer<DTYPE> buf3(buf3_ptr, W);
    Halide::Runtime::Buffer<DTYPE> buf4(buf4_ptr, W);
    Halide::Runtime::Buffer<DTYPE> buf5(buf5_ptr, W);
#endif

    // buf0(x) = cast<DTYPE>(11);
    // buf0(lut1_update) = input_vtcm(r.x); // ---> Scatter
    // buf0(r.x) = input_vtcm(lut1_update); // ---> Gather
    // buf_out0(x) = buf0(x); // ---> Load
    for (int x = 0; x < W; x++) {
        buf0(x) = input(lut1(x));
    }

    // buf1(x) = input_vtcm(lut1_pure); // ---> Gather
    // buf1(r.x) = input_vtcm(lut2_update); // ---> Gather
    // buf_out1(x) = buf1(x); // ---> Load
    for (int x = 0; x < W; x++) {
        buf1(x) = input(lut2(x));
    }

    // buf2(x) = input_vtcm(lut1_pure); // ---> Gather
    for (int x = 0; x < W; x++) {
        buf2(x) = input(lut1(x));
    }
    // buf2(lut2_update) = input_vtcm(r.x) * 7; // ---> Scatter
    // buf_out2(x) = buf2(x); // ---> Load
    for (int x = 0; x < W; x++) {
        buf2(lut2(x)) = input(x) * 2;
    }

    // buf3(x) = cast<DTYPE>(61);
    for (int x = 0; x < W; x++) {
        buf3(x) = static_cast<DTYPE>(61);
    }
    // buf3(lut2_update) = input_vtcm(r.x); // ---> Scatter
    for (int x = 0; x < W; x++) {
        buf3(lut2(x)) = input(x);
    }
    // buf3(lut1_update) = cast<DTYPE>(41); // ---> Scatter
    // buf_out3(x) = buf3(x); // ---> Load
    for (int x = 0; x < W; x++) {
        buf3(lut1(x)) = static_cast<DTYPE>(41);
    }

    // buf_out4(x) = cast<DTYPE>(11);
    // buf_out4(lut2_update) = input_vtcm(r.x) * 23; // ---> Scatter
    // buf_out4(r.x) = input_vtcm(r.x); // ---> Store
    for (int x = 0; x < W; x++) {
        buf4(x) = input(x);
    }

    // buf_out5(x) = input_vtcm(lut1_pure); // ---> Gather
    // buf_out5(r.x) = cast<DTYPE>(97); // ---> Store
    for (int x = 0; x < W; x++) {
        buf5(x) = static_cast<DTYPE>(97);
    }

    for (int x = 0; x < W; x++) {
        int ref = /*buf0(x) + buf1(x) + buf2(x) + */buf3(x) + buf4(x)/* + buf5(x)*/;
        // printf("(%4d): %3d (Halide) == %3d (Expected)\n",
        //         x, output(x), ref);
        if (output(x) != ref) {
            errcnt++;
            // if (errcnt <= maxerr) {
                printf("Mismatch at (%4d): %3d (Halide) != %3d (Expected)\n",
                        x, output(x), ref);
            // }
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
    Halide::Runtime::Buffer<DTYPE>   in(nullptr, W);
    Halide::Runtime::Buffer<DTYPE> lut1(nullptr, W);
    Halide::Runtime::Buffer<DTYPE> lut2(nullptr, W);
    Halide::Runtime::Buffer<DTYPE>  out(nullptr, W);

    in.device_malloc(halide_hexagon_device_interface());
    lut1.device_malloc(halide_hexagon_device_interface());
    lut2.device_malloc(halide_hexagon_device_interface());
    out.device_malloc(halide_hexagon_device_interface());
#else
    DTYPE *in_ptr   = (DTYPE *)memalign(1 << LOG2VLEN, W*sizeof(DTYPE));
    DTYPE *lut1_ptr = (DTYPE *)memalign(1 << LOG2VLEN, W*sizeof(DTYPE));
    DTYPE *lut2_ptr = (DTYPE *)memalign(1 << LOG2VLEN, W*sizeof(DTYPE));
    DTYPE *out_ptr  = (DTYPE *)memalign(1 << LOG2VLEN, W*sizeof(DTYPE));

    Halide::Runtime::Buffer<DTYPE>   in(in_ptr , W);
    Halide::Runtime::Buffer<DTYPE> lut1(in_ptr , W);
    Halide::Runtime::Buffer<DTYPE> lut2(in_ptr , W);
    Halide::Runtime::Buffer<DTYPE>  out(out_ptr, W);
#endif

    srand(0);
    // Fill the input image
    in.for_each_value([&](DTYPE &x) {
        x = static_cast<DTYPE>(rand()) % W;
    });
    for(int i = 0; i < W; i++) {
        in(i) = static_cast<DTYPE>(rand()) % W;
        lut1(i) = static_cast<DTYPE>(i);
        lut2(i) = static_cast<DTYPE>(i);
    }
    // Create a random permutation for lut1 and lut2. All indices should be
    // unique for scatters to avoid race conditions inside scatters.
    for(int i = 0; i < 1000; i++) {
        swap(lut1, rand() % W, rand() % W);
        swap(lut2, rand() % W, rand() % W);
    }
#ifdef HL_HEXAGON_DEVICE
    // To avoid the cost of powering HVX on in each call of the
    // pipeline, power it on once now. Also, set Hexagon performance to turbo.
    halide_hexagon_set_performance_mode(NULL, halide_hexagon_power_turbo);
    halide_hexagon_power_hvx_on(NULL);
#endif

    printf("Image size: %d pixels\n", W);
    printf("Histogram element size: %d bits\n", (int)sizeof(DTYPE));
    printf("Running pipeline...\n");

#ifdef HL_HEXAGON_DEVICE
    double time = Halide::Tools::benchmark(iterations, 1, [&]() {
#else
    double time = benchmark([&]() {
#endif
    int result = pipeline(in, lut1, lut2, out);
        if (result != 0) {
            printf("pipeline failed! %d\n", result);
        }
    });
    out.copy_to_host();
    printf("Done, TIME: %g ms\nTHROUGHPUT: %g MP/s\n", time*1000.0, ((double)(W))/((double)(1000000) * time));

#ifdef HL_HEXAGON_DEVICE
    // We're done with HVX, power it off, and reset the performance mode
    // to default to save power.
    halide_hexagon_power_hvx_off(NULL);
    halide_hexagon_set_performance_mode(NULL, halide_hexagon_power_default);
#endif

#if 1
    if (checker(in, lut1, lut2, out) != 0) {
        printf("Fail!\n");
        return 1;
    }
#endif
    printf("Success!\n\n");

    return 0;
}
