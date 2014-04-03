#include "typesystem.hpp"
#include "MurmurHash3.h"    // public domain
#include <sstream>
#include <limits>

enum {HASHSEED = 0xabcdef};

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

    uint32_t hash;
    MurmurHash3_x86_32(&pair, sizeof(pair), HASHSEED, &hash);
    return hash;
}

unsigned int hash(const Type* key) {
    struct {
        const Type* ptr;
    } raw;

    raw.ptr = key;

    uint32_t hash;
    MurmurHash3_x86_32(&raw, sizeof(raw), HASHSEED, &hash);
    return hash;
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

void canPromote(TypeContext &ctx, std::string from, std::string to) {
    ctx.setCompatibility(ctx.types.get(from),
                         ctx.types.get(to),
                         TCC_PROMOTE);
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

    // Set float promotion
    canPromote(ctx, "float32", "float64");

    // Set complex promotion
    canPromote(ctx, "complex64", "complex128");

    // int8, int16 can promote to float32
    canPromote(ctx, "int8", "float32");
    canPromote(ctx, "int16", "float32");
    canPromote(ctx, "uint8", "float32");
    canPromote(ctx, "uint16", "float32");

    // int32 can promote to float64
    canPromote(ctx, "int32", "float64");
    canPromote(ctx, "uint32", "float64");
}

struct type_lessor {
    TypeContext &ctx;

    type_lessor(TypeContext &ctx): ctx(ctx) { }

    bool operator () (Type* a, Type *b) {
        int diff = ctx.getRank(a) - ctx.getRank(b);
        return diff < 0;
    }
};


CoerceDescriptor coerce(TypeContext &ctx, Type** typeset, size_t n){
    // Prefer a type that all other types can promote to.
    CoerceDescriptor howcoerce;

    for(size_t i = 0; i < n; ++i) {
        size_t goodct = 0;
        for(size_t j = 0; j < n; ++j) {
            CastDescriptor howcast = ctx.cast(typeset[i], typeset[j]);
            if (howcast.tcc == TCC_EXACT || howcast.tcc == TCC_PROMOTE) {
                goodct += 1;
            } else if (howcast.tcc == TCC_FALSE) {
                return howcoerce;
            }
        }

        if (goodct == n) {
            howcoerce.okay = true;
            howcoerce.safe = true;
            howcoerce.type = typeset[i];
        }
    }

    if (howcoerce.okay) return howcoerce;

    // Otherwise use the type with the highest rank if they can be converted
    howcoerce.okay = true;
    howcoerce.safe = false;
    howcoerce.type = *std::max_element(typeset, typeset + n, type_lessor(ctx));
    return howcoerce;
}

std::string explainCoerce(CoerceDescriptor cd){
    if (!cd.okay) return "coercion is impossible";
    std::ostringstream oss;
    oss << (cd.safe? "safe" : "unsafe") << " coerce to " << cd.type->name;
    return oss.str();
}


struct Rating{
    unsigned short promote;
    unsigned short convert;

    Rating(): promote(0), convert(0) {}

    void bad() {
        convert = promote = std::numeric_limits<unsigned short>::max();
    }

    bool operator < (const Rating &other) const {
        unsigned short self[] = {convert, promote};
        unsigned short that[] = {other.convert, other.promote};
        for(unsigned i = 0; i < sizeof(self)/sizeof(self[0]); ++i) {
            if (self[i] < that[i]) return true;
        }
        return false;
    }

    bool operator == (const Rating &other) const {
        return promote == other.promote && convert == other.convert;
    }
};


int selectOverload(TypeContext &ctx, Type* sig[], Type* overloads[],
                   int selected[], int nvers, int nargs, Rating ratings[])
{
    // Symmetric overload resolution
    // This is like C++ overload resolution

    // Rate each version
    int badct = 0;
    for (int i = 0; i < nvers; ++i) {
        Type **entry = &overloads[i * nargs];
        Rating &rate = ratings[i];

        for(int j = 0; j < nargs; ++j) {
            CastDescriptor desc = ctx.cast(entry[j], sig[j]);
            if (desc.tcc == TCC_FALSE) {
                rate.bad();
                badct += 1;
                break;
            }
            switch(desc.tcc) {
            case TCC_PROMOTE:
                rate.promote += 1;
                break;
            case TCC_CONVERT:
                rate.convert += 1;
                break;
            default:
                break;
            }
        }
    }

    // No match?
    if (badct == nvers) return 0;

    // Find the best rating and safe all possible combination.
    Rating best;
    best.bad();

    int matchct = 0;
    int *selptr = selected;
    for (int i = 0; i < nvers; ++i) {
        if (ratings[i] < best) {
            // Found a new best
            best = ratings[i];
            // Reset counters
            matchct = 1;
            selptr = selected;
            *selptr++ = i;
        } else if (ratings[i] == best) {
            matchct += 1;
            *selptr++ = i;
        }
    }
    return matchct;
}

int selectOverload(TypeContext &ctx, Type* sig[], Type* overloads[],
                   int selected[], int nvers, int nargs)
{
    if (nvers < 16) {
        Rating ratings[16];
        return selectOverload(ctx, sig, overloads, selected, nvers, nargs,
                              ratings);
    } else {
        Rating *ratings = new Rating[nvers];
        int result = selectOverload(ctx, sig, overloads, selected, nvers,
                                    nargs, ratings);
        delete [] ratings;
        return result;
    }
}


int compareCast(CastDescriptor a, CastDescriptor b) {
    int tmp = a.tcc - b.tcc;
    if (tmp == 0) {
        tmp = a.distance - b.distance;
    }
    return tmp;
}


int selectBestOverload(TypeContext &ctx, Type* sig[], Type* overloads[],
                       int selected[], int nvers, int nargs)
{
    int ct = selectOverload(ctx, sig, overloads, selected, nvers, nargs);
    if (ct <= 0) return -1;
    if (ct == 1) return selected[0];

    // Otherwise perform conflict resolution
    int samect = 0;
    int *selptr = selected;
    CastDescriptor bestcast;

    // Select asymmetrically with the left most argument being the most
    // important
    for(int j = 0; j < nargs; ++j) {
        // Collect cast description
        int i = 0;
        CastDescriptor cd = ctx.cast(overloads[selected[i] * nargs + j],
                                     sig[j]);
        for(i = 1; i < ct; ++i) {
            int cmp = compareCast(cd, bestcast);
            if (cmp < 0) {
                bestcast = cd;
                selptr = selected;
                samect = 1;
                *selptr++ = i;
            } else if (cmp == 0){
                samect += 1;
                *selptr++ = i;
            }
        }

        ct = samect;
        if (ct == 1) return selected[0];
    }
    return ct;
}

int selectBestOverload(TypeContext &ctx, Type* sig[], Type* overloads[],
                       int nvers, int nargs)
{
    if (nvers < 16) {
        int selected[16];
        return selectBestOverload(ctx, sig, overloads, selected, nvers, nargs);
    } else {
        int *selected = new int[nvers];
        int result = selectBestOverload(ctx, sig, overloads, selected, nvers,
                                        nargs);
        delete [] selected;
        return result;
    }
}





} // namespace numba
