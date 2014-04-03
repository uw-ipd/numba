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

    Type* typeset[3] = {ctx.types.get("int32"),
                        ctx.types.get("int64"),
                        ctx.types.get("boolean")};

    std::cout << explainCoerce(coerce(ctx, typeset, 3)) << '\n';



    Type* sig[] = {ctx.types.get("int32"), ctx.types.get("float32")};
    Type* vers[] = {ctx.types.get("float32"), ctx.types.get("float32"),
                    ctx.types.get("int32"), ctx.types.get("int32"),
                    ctx.types.get("complex64"), ctx.types.get("complex64")};

    int nargs = sizeof(sig) / sizeof(sig[0]);
    int nvers = sizeof(vers) / sizeof(sig);
    int selected[nvers];

    int ct = selectOverload(ctx, sig, vers, selected, nvers, nargs);
    std::cout << "selected " << ct << '\n';
    for (int i = 0; i < ct; ++i) {
        std::cout << "i = " << i << " | " << selected[i] << '\n';
    }

    std::cout << "best "
              << selectBestOverload(ctx, sig, vers, nvers, nargs)
              << '\n';

    return 0;
}
