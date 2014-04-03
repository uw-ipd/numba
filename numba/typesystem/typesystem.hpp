#ifndef NUMBA_TYPESYSTEM_H_
#define NUMBA_TYPESYSTEM_H_

#include <map>
#include <set>
#include <vector>
#include <string>
#include <algorithm>

namespace numba {

/*
Some symbols that are

*/
struct Atom {
    std::string name;
};

typedef std::set<const Atom*> atom_set_type;

struct Trait : Atom { };

struct Type : Atom {
    atom_set_type traits;

    bool hasTrait(const Trait *trait) const;
    void addTrait(const Trait *trait);
};

template<class T>
class AtomContext {
public:
    typedef std::map<std::string, T*> atom_map_type;

    ~AtomContext() {
        typename atom_map_type::iterator it;
        for(it = theMap.begin(); it != theMap.end(); ++it) {
            delete it->second;
        }
    }

    T* get(const std::string &name) {
        typename atom_map_type::iterator it = theMap.find(name);
        if (it == theMap.end()){
            T *atom = new T;
            theMap[name] = atom;
            atom->name = name;
            return atom;
        } else {
            return static_cast<T*>(it->second);
        }
    }

private:
    atom_map_type theMap;
};


enum TypeCompatibleCode{
	// No match
	TCC_FALSE = 0,
	// Exact match
	TCC_EXACT,
	// Promotion with no precision loss
	TCC_PROMOTE,
	// Conversion with precision loss
	// e.g. int32 to double
    TCC_CONVERT,
    // Subtype is UNUSED
    //	TCC_SUBTYPE,
};



template<class Tkey, class Tval, size_t Tsize>
class HashMap {
public:
    typedef std::pair<Tkey, Tval> record_type;
    typedef std::vector<record_type> hash_bin_type;
    typedef std::pair<bool, Tval> find_type;

    HashMap(): count(0) {}

    /*
    Warning: should not have duplicated key
    */
    void insert(Tkey key, Tval val) {
        unsigned int i = hash(key) % Tsize;
        hash_bin_type &bin = records[i];
        record_type data;
        data.first = key;
        data.second = val;
        for (size_t j = 0; j < bin.size(); ++j) {
            if (bin[j].first == key) {
                bin[j].second = val;
                return;
            }
        }
        bin.push_back(data);
        // Sort the bin
        // Allow for slow insert but fast search
        std::sort(bin.begin(), bin.end());
        count += 1;
    }

    find_type find(Tkey key) const {
        unsigned int i = hash(key) % Tsize;
        const hash_bin_type &bin = records[i];
        // Assume the bin the sorted
        for (size_t j = 0; j < bin.size(); ++j) {
            if (bin[j].first == key) {
                return std::make_pair(true, bin[j].second);
            } else if (bin[j].first > key) {
                break;
            }
        }
        return std::make_pair(false, Tval());
    }

    size_t size() const { return count; }
private:
    hash_bin_type records[Tsize];
    size_t count;
};

typedef std::pair<const Type*, const Type*> TypePair;
typedef HashMap<TypePair, TypeCompatibleCode, 199> TCCMap;
typedef HashMap<const Type*, int, 199> RankMap;

unsigned int hash(const TypePair &key);
unsigned int hash(const Type* key);


struct CastDescriptor{
    TypeCompatibleCode tcc;
    int                distance;    // only use for TCC_CONVERT
};


/*
Types and Traits are built and owned from a TypeContext.
TypeContext should be used as a singleton object.
*/
class TypeContext {
public:
    AtomContext<Trait> traits;
    AtomContext<Type> types;

    // type compatibility
    void setCompatibility(const Type *from, const Type *to,
                          TypeCompatibleCode tcc);

    TypeCompatibleCode getCompatibility(const Type *from,
                                        const Type *to) const;

    std::string explainCompatibility(const Type *from, const Type *to) const;

    // type ranking for machines types
    void appendRank(const Type* type);
    int getRank(const Type* type) const;

    CastDescriptor cast(const Type *from, const Type *to) const;
    std::string explainCast(const Type *from, const Type *to) const;
    std::string explainCast(std::string from, std::string to);

private:
    TCCMap castRules;
    RankMap ranking;
};

std::string explainCompatibility(TypeCompatibleCode tcc);

void fillMachineTypes(TypeContext &ctx);

struct CoerceDescriptor{
    bool okay;
    bool safe;
    Type *type;
};

CoerceDescriptor coerce(TypeContext &ctx, Type** typeset, size_t n);
CoerceDescriptor coerce(TypeContext &ctx, Type** typeset, size_t n);

std::string explainCoerce(CoerceDescriptor cd);

/*
Select compatible overload versions.
*/
int selectOverload(TypeContext &ctx, Type* sig[], Type* overloads[],
                   int selected[], int nvers, int nargs);

/*
Select the best overload version with asymmetric resolution (left to right).
*/
int selectBestOverload(TypeContext &ctx, Type* sig[], Type* overloads[],
                       int nvers, int nargs);


} // namespace numba

#endif  // NUMBA_TYPESYSTEM_H_
