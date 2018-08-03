#include "Halide.h"

using namespace Halide;

// Implements a simple gather pipeline to make use of VTCM available on v65+
// hexagon DSP.
template<typename ITYPE>
int test() {
    const int W = 128;
    const int H = 2;

    Buffer<ITYPE> input(W+1, H);
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W+1; x++) {
            input(x, y) = (ITYPE)(1);
        }
    }

    Buffer<ITYPE> lut(W, H);
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            lut(x, y) = (ITYPE)(2);
        }
    }

    Var x, y;
    Func lut_vtcm, gather_vtcm, gather;

    Expr xCoord = clamp(cast<int>(input(x, y)), 0, W-1);
    Expr yCoord = clamp(cast<int>(input(x+1, y)), 0, H-1);
    lut_vtcm(x, y) = lut(x, y);
    gather_vtcm(x, y) = lut_vtcm(xCoord, yCoord);
    gather(x, y) = gather_vtcm(x, y);

    Target target = get_jit_target_from_environment();
    if (target.features_any_of({Target::HVX_64, Target::HVX_128})) {
        // lut_vtcm
            // .compute_at(gather, y)
            // .store_in(MemoryType::Vtcm)
            // .vectorize(x, 64);

        // gather_vtcm
            // .compute_at(gather, y)
            // .store_in(MemoryType::Vtcm)
            // .vectorize(x, 64);

        gather
            .hexagon()
            .vectorize(x, 128);
    }

    Buffer<ITYPE> output = gather.realize(W, H);

    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            int xCoord = std::max(std::min((int)(input(x, y)), W-1), 0);
            int yCoord = std::max(std::min((int)(input(x+1, y)), H-1), 0);
            ITYPE correct = lut(xCoord, yCoord);
            if (output(x, y) != correct) {
                printf("output(%d, %d) = %d instead of %d\n", x, y, output(x, y), correct);
                return false;
            }
        }
    }

    return true;
}

int main() {
    if (!test<uint16_t>()/* && !test<uint32_t>()*/) return 1;
    printf("Success!\n");
    return 0;
}
