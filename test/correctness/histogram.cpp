#include <stdio.h>
#include "Halide.h"

using namespace Halide;

template<typename ITYPE, typename HTYPE>
bool test() {
    Target target = get_jit_target_from_environment();
    int W = 128, H = 128;

    // Compute a random image and its true histogram
    HTYPE reference_hist[256];
    for (int i = 0; i < 256; i++) {
        reference_hist[i] = HTYPE(0);
    }

    Buffer<ITYPE> in(W, H);
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            // in(x, y) = ITYPE(rand() & 0x000000ff);
            in(x, y) = ITYPE(x & 0x000000ff);
            reference_hist[uint8_t(in(x, y))] += HTYPE(1);
        }
    }

    Func hist("hist");
    Var x;

    RDom r(in);
    hist(x) = HTYPE(0);
    hist(clamp(cast<int>(in(r.x, r.y)), 0, 255)) += HTYPE(1);

    hist.compute_root();

    Func g;
    g(x) = hist(x+10);

    // No parallel reductions
    /*
    Target target = get_jit_target_from_environment();
    if (target.has_gpu_feature()) {
        Var xi;
        hist.gpu_tile(x, xi, 64);
        RVar rxi, ryi;
        hist.update().gpu_tile(r.x, r.y, rxi, ryi, 16, 16);
    }
    */

    if (target.features_any_of({Target::HVX_64, Target::HVX_128})) {
        const int vector_size = target.has_feature(Target::HVX_128) ? 128 : 64;
        Var yi;

        g
            .hexagon()
            .vectorize(x, vector_size/2);

        hist
            .compute_at(g, Var::outermost())
            .store_in(MemoryType::VTCM)
            .vectorize(x, vector_size/2);

        hist
            .update(0)
            .allow_race_conditions()
            .vectorize(r.x, vector_size/2);
    }

    Buffer<int32_t> histogram = g.realize(10); // buckets 10-20 only

    for (int i = 10; i < 20; i++) {
        if (histogram(i-10) != reference_hist[i]) {
            printf("Error: bucket %d is %d instead of %d\n", i, histogram(i), reference_hist[i]);
            return false;
        }
    }

    return true;
}

int main(int argc, char **argv) {
    if (!test<uint8_t, int>()) return 1;
    // if (!test<float, int>()) return 1;
    printf("Success!\n");
    return 0;
}
