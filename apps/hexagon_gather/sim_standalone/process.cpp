#include <hexagon_standalone.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include "pipeline.h"
#include "process.h"
#include "HalideBuffer.h"

// Verify result for the halide pipeline
int checker(Halide::Runtime::Buffer<DTYPE> &in,
        Halide::Runtime::Buffer<DTYPE> &lut,
        Halide::Runtime::Buffer<DTYPE> &out) {
    int errcnt = 0, maxerr = 10;
    printf("Checking...\n");

    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            int xidx = std::min(std::max((int)in(x, 0), 0), W-1);
            int yidx = std::min(std::max((int)in(x, 1), 0), H-1);
            if (out(x, y) != lut(xidx, yidx)) {
                errcnt++;
                // if (errcnt <= maxerr) {
                    printf("Mismatch at (%d, %d): %d (Halide) == %d (Expected)\n",
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

int upround(int x) {
  return (x + VLEN - 1) & (-VLEN);
}

int main(int argc, char **argv) {
  // Create the Input.
  constexpr int dims = 2;

  DTYPE *input  = (DTYPE *)memalign(1 << LOG2VLEN, upround(W)*H*sizeof(DTYPE));
  DTYPE *lut    = (DTYPE *)memalign(1 << LOG2VLEN, upround(W)*H*sizeof(DTYPE));
  DTYPE *output = (DTYPE *)memalign(1 << LOG2VLEN, upround(W)*H*sizeof(DTYPE));

  if (!input || !lut || !output) {
    printf("Error: Could not allocate Memory for image\n");
    return 1;
  }

  halide_dimension_t x_dim1{0, W, 1};
  halide_dimension_t y_dim1{0, H, upround(W)};
  halide_dimension_t io_shape1[2] = {x_dim1, y_dim1};

  halide_dimension_t x_dim2{0, W, 1};
  halide_dimension_t y_dim2{0, H, upround(W)};
  halide_dimension_t io_shape2[2] = {x_dim2, y_dim2};

  halide_dimension_t x_dim3{0, W, 1};
  halide_dimension_t y_dim3{0, H, W};
  halide_dimension_t io_shape3[2] = {x_dim3, y_dim3};

  Halide::Runtime::Buffer<DTYPE> input_buf(input, dims, io_shape1);
  Halide::Runtime::Buffer<DTYPE> lut_buf(lut, dims, io_shape2);
  Halide::Runtime::Buffer<DTYPE> output_buf(output, dims, io_shape3);

  // srand(time(0));
  for (int x = 0; x < W; x++) {
    DTYPE idx = static_cast<DTYPE>(rand());
    input_buf(x, 0) = idx % W;
    input_buf(x, 1) = idx % 2;
  }

  for (int y = 0; y < H; y++)
    for (int x = 0; x < W; x++) {
      lut_buf(x, y) = static_cast<DTYPE>(rand());
      // printf("Lut %d\n", lut_buf(x, y));
    }

  // Run in 128 byte mode
  SIM_ACQUIRE_HVX;
  SIM_SET_HVX_DOUBLE_MODE;
    int error = pipeline(input_buf, lut_buf, output_buf);
    if (error != 0) {
      printf("Pipeline failed: %d\n", error);
      abort();
    }
  SIM_RELEASE_HVX;

#if 1
    if (checker(input_buf, lut_buf, output_buf) != 0) {
        printf("Fail!\n");
        return 1;
    }
#endif
    printf("Success!\n");
  return 0;
}
