#include "Halide.h"
#include <stdint.h>
#include "div.h"

using namespace Halide;

class Div : public Halide::Generator<Div> {

public:
    Input<Buffer<DTYPE>> num{"num", 2};
    Input<Buffer<DTYPE>> den{"den", 2};
    Output<Buffer<DTYPE>> res{"res", 2};

    void generate() {
#ifdef SCALAR
        res(x, y) = num(x, y) / den(x, y);
#else
        res(x, y) = long_div(num, den)(x, y); 
#endif
    std::cout << "Runing: " << std::endl;
    }

    void schedule() {
        if (get_target().features_any_of({Target::HVX_64, Target::HVX_128})) {
            const int vector_size = get_target().has_feature(Target::HVX_128) ? 128 : 64;

            Expr num_stride = num.dim(1).stride();
            Expr den_stride = den.dim(1).stride();
            Expr res_stride = res.dim(1).stride();
            num.dim(1).set_stride((num_stride/vector_size) * vector_size);
            den.dim(1).set_stride((den_stride/vector_size) * vector_size);
            res.dim(1).set_stride((res_stride/vector_size) * vector_size);
            num.dim(0).set_min(0);
            num.dim(1).set_min(0);
            den.dim(0).set_min(0);
            den.dim(1).set_min(0);
            res.dim(0).set_min(0);
            res.dim(1).set_min(0);

            num.set_host_alignment(vector_size);
            den.set_host_alignment(vector_size);
            res.set_host_alignment(vector_size);

#ifdef SCALAR
            res.hexagon()
                .split(y, yp, y, 64)
                .parallel(yp)
                .prefetch(num, y, 2)
                .prefetch(den, y, 2)
                .vectorize(x, vector_size)
                ;
#else
            res.hexagon()
                .split(y, yp, y, 64)
                .parallel(yp)
                .prefetch(num, y, 2)
                .prefetch(den, y, 2)
                .vectorize(x, vector_size)
                ;
#endif
        }
    }

private:
    Func long_div(Func num, Func den);
    Var x{"x"}, y{"y"}, yp{"yp"};
};

Func Div::long_div(Func num, Func den) {
    Var x{"x"}, y{"y"};
    const int times = sizeof(DTYPE) * 8 + 1;
    Func num_32[times], leading_zeros{"leading_zeros"}, q[times], curr[times];
    num_32[0](x, y) = num(x, y);
    q[0](x, y) = cast<DTYPE>(0);

    leading_zeros(x, y) = times - 1 - count_leading_zeros(den(x, y));
    for (int i = 1; i < times; i++) {
        curr[i](x, y) = num_32[i-1](x, y) >> leading_zeros(x, y);
        q[i](x, y) = q[i-1](x, y) + curr[i](x, y);
        // The following can be proven:
        // 1. count_leading_zeros(curr[i](x, y) * den(x, y)) == count_leading_zeros(num_32[i-1](x, y))
        // 2. curr[i](x, y) * den(x, y) <= num_32[i-1](x, y)
        num_32[i](x, y) = num_32[i-1](x, y) - curr[i](x, y) * den(x, y);
    }
    Func final{"final"};
    final(x, y) = cast<DTYPE>(select(num_32[times-1](x, y) >= den(x, y), q[times-1](x, y)+1, q[times-1](x, y)));
    return final;
}


HALIDE_REGISTER_GENERATOR(Div, fast_div);

