#include <iostream>
#include "typesystem.hpp"

int main(){
    using namespace numba;
    TypeContext ctx;
    fillMachineTypes(ctx);
    std::cout << ctx.explainCast("int32", "float32") << '\n';
    std::cout << ctx.explainCast("int32", "int32") << '\n';
    std::cout << ctx.explainCast("int32", "int64") << '\n';
    std::cout << ctx.explainCast("int64", "int16") << '\n';
    std::cout << ctx.explainCast("uint32", "uint64") << '\n';
    std::cout << ctx.explainCast("uint32", "int64") << '\n';
    std::cout << ctx.explainCast("int64", "uint16") << '\n';
    return 0;
}
