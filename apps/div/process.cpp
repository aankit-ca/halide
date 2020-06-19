#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <time.h>
#include "halide_benchmark.h"
#include "div_pipeline.h"
#include "HalideRuntimeHexagonHost.h"
#include "HalideBuffer.h"
#include "div.h"

using namespace Halide::Runtime;
using namespace Halide::Tools;

// Verify result for the halide pipeline
int checker(Buffer<DTYPE> &num, Buffer<DTYPE> &den, Buffer<DTYPE> &res) {
    // Algorithm
    for (int y = 0; y < HEIGHT; y++) {
        for(int x = 0; x < WIDTH; x++) {
            DTYPE expected = num(x, y) / den(x, y);
            if (expected != res(x, y)) {
                printf("Mismatch at (%d, %d) (%d / %d) :  %d (Halide) != %d (Expected);\n",
                              x, y, num(x, y), den(x, y), res(x, y), expected);
                return 0;
            }
        }
    }
    return 1;
}

int main(int argc, char **argv) {

    int iterations = ITERATIONS;
    int width_128 = (WIDTH + VLEN -1)&(-VLEN);

    constexpr int dims = 2;
    halide_dimension_t x_dim{0, WIDTH, 1};
    halide_dimension_t y_dim{0, HEIGHT, width_128};
    halide_dimension_t io_shape[2] = {x_dim, y_dim};

    Halide::Runtime::Buffer<DTYPE> num(nullptr, dims, io_shape);
    Halide::Runtime::Buffer<DTYPE> den(nullptr, dims, io_shape);
    Halide::Runtime::Buffer<DTYPE> res(nullptr, dims, io_shape);
    num.device_malloc(halide_hexagon_device_interface());
    den.device_malloc(halide_hexagon_device_interface());
    res.device_malloc(halide_hexagon_device_interface());

    srand (time(NULL));

    // Fill the input buffer with random data. No required though.
    const DTYPE max_val = std::numeric_limits<DTYPE>::max();
    num.for_each_value([&](DTYPE &x) {
        x = static_cast<DTYPE>(rand() % max_val);
    });
    // Fill the input buffer with random data. No required though.
    den.for_each_value([&](DTYPE &x) {
        x = static_cast<DTYPE>(rand() % max_val + 1);
    });

    // To avoid the cost of powering HVX on in each call of the
    // pipeline, power it on once now. Also, set Hexagon performance to turbo.
    halide_hexagon_set_performance_mode(NULL, halide_hexagon_power_turbo);
    halide_hexagon_power_hvx_on(NULL);

    printf("Running pipeline...\n");
    double time = Halide::Tools::benchmark(iterations, ITERATIONS, [&]() {
        int result = div_pipeline(num, den, res);
        if (result != 0) {
            printf("pipeline failed! %d\n", result);
        }
    });
    res.copy_to_host();

    printf("Done, TIME: %g ms\nTHROUGHPUT: %g MP/s\n", time*1000.0, ((double)(WIDTH * HEIGHT))/((double)(1000000) * time));

    // We're done with HVX, power it off, and reset the performance mode
    // to default to save power.
    halide_hexagon_power_hvx_off(NULL);
    halide_hexagon_set_performance_mode(NULL, halide_hexagon_power_default);
    halide_profiler_report(nullptr);

    if (!checker(num, den, res)) {
        printf("fast_div Failed\n");
        // abort();
    } else {
        printf("fast_div Passed\n");
        printf("Success!\n");
    }

    return 0;
}
