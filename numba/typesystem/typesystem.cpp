#include "typesystem.hpp"
#include <sstream>

// Uses MIT licensed Murmur Hash
unsigned int MurmurHash2(const void * key, int len, unsigned int seed);

namespace numba {

bool Type::hasTrait(const Trait *trait) const {
    return traits.end() != traits.find(trait);
}

void Type::addTrait(const Trait *trait) {
    traits.insert(trait);
}

unsigned int hash(const TypePair &key) {
    struct {
        size_t a, b;
    } pair;

    pair.a = reinterpret_cast<size_t>(key.first);
    pair.b = reinterpret_cast<size_t>(key.second);

    return MurmurHash2(&pair, sizeof(pair), 0xabcdef);
}

unsigned int hash(const Type* key) {
    struct {
        const Type* ptr;
    } raw;

    raw.ptr = key;

    return MurmurHash2(&raw, sizeof(raw), 0xabcdef);
}

void
TypeContext::setCompatibility(const Type *from, const Type *to,
                              TypeCompatibleCode tcc) {
    castRules.insert(TypePair(from, to), tcc);
}

TypeCompatibleCode
TypeContext::getCompatibility(const Type *from, const Type *to) const {
    TCCMap::find_type result = castRules.find(TypePair(from, to));
    if (result.first){
        return result.second;
    } else {
        return TCC_FALSE;
    }
}

std::string
TypeContext::explainCompatibility(const Type *from, const Type *to) const {
    return numba::explainCompatibility(getCompatibility(from, to));
}

void TypeContext::appendRank(const Type* type) {
    ranking.insert(type, ranking.size());
}

int TypeContext::getRank(const Type* type) const {
    RankMap::find_type result = ranking.find(type);
    if (result.first) {
        return result.second;
    } else {
        return 0;
    }
}

CastDescriptor TypeContext::cast(const Type *from, const Type *to) const {

    CastDescriptor cd;
    if (from == to) {
        cd.tcc = TCC_EXACT;
    } else {
        cd.tcc = getCompatibility(from, to);
    }

    if (cd.tcc == TCC_CONVERT) {
        cd.distance = getRank(to) - getRank(from);
    }
    return cd;
}

std::string TypeContext::explainCast(const Type *from, const Type *to) const {
    std::ostringstream oss;
    oss << from->name << "->" << to->name << " :: ";

    CastDescriptor cd = cast(from, to);
    oss << numba::explainCompatibility(cd.tcc);

    if (cd.tcc == TCC_CONVERT) {
        oss << ":";
        if (cd.distance > 0)
            oss << "up";
        else if (cd.distance < 0)
            oss << "down";
        else
            oss << "invalid";
    }
    return oss.str();
}

std::string TypeContext::explainCast(std::string from, std::string to) {
    return explainCast(types.get(from), types.get(to));
}

std::string explainCompatibility(TypeCompatibleCode tcc) {
    switch (tcc){
    case TCC_FALSE:
        return "false";
    case TCC_EXACT:
        return "exact";
    case TCC_PROMOTE:
        return "promote";
    case TCC_CONVERT:
        return "convert";
    }
}


const char MachineTypes[][12] = {
"boolean",
"uint8", "int8",
"uint16", "int16",
"uint32", "int32",
"uint64", "int64",
"float32",
"float64",
"complex64",
"complex128",
};


void fillIntegerRules(TypeContext &ctx, std::string prefix) {
    const unsigned int bits[] = {8, 16, 32, 64};
    const size_t NBITS = sizeof(bits) / sizeof(bits[0]);

    for(size_t i = 0; i < NBITS; ++i) {
        std::ostringstream oss;
        oss << prefix << bits[i];
        Type *ti = ctx.types.get(oss.str());
        for (size_t j = i + 1; j < NBITS; ++j) {
            std::ostringstream oss;
            oss << prefix << bits[j];
            Type *tj = ctx.types.get(oss.str());
            ctx.setCompatibility(ti, tj, TCC_PROMOTE);
        }
    }
}

void fillMachineTypes(TypeContext &ctx) {
    const size_t N = sizeof(MachineTypes) / sizeof(MachineTypes[0]);

    // Initialize types and their ranks
    for (size_t i = 0; i < N; ++i) {
        ctx.appendRank(ctx.types.get(MachineTypes[i]));
    }

    // Initialize type casting rule to conversion
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            Type *ti = ctx.types.get(MachineTypes[i]);
            Type *tj = ctx.types.get(MachineTypes[j]);
            if (ti != tj) {
                ctx.setCompatibility(ti, tj, TCC_CONVERT);
            }
        }
    }

    // Set all integer promotion
    fillIntegerRules(ctx, "uint");
    fillIntegerRules(ctx, "int");
}



} // namespace numba
