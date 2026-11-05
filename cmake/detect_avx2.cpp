// Probe used by CMake (try_run) to discover whether the *build host* supports AVX2.
// Exit code 0 => AVX2 + BMI2 + POPCNT available.
#if defined(_MSC_VER)
#include <intrin.h>
static bool has_avx2() {
    int info[4];
    __cpuid(info, 0);
    if (info[0] < 7) return false;
    __cpuid(info, 1);
    const bool osxsave = (info[2] & (1 << 27)) != 0;
    const bool avx = (info[2] & (1 << 28)) != 0;
    const bool popcnt = (info[2] & (1 << 23)) != 0;
    if (!osxsave || !avx || !popcnt) return false;
    if ((_xgetbv(0) & 0x6) != 0x6) return false; // OS saves YMM state
    __cpuidex(info, 7, 0);
    const bool avx2 = (info[1] & (1 << 5)) != 0;
    const bool bmi2 = (info[1] & (1 << 8)) != 0;
    return avx2 && bmi2;
}
#elif defined(__GNUC__) || defined(__clang__)
static bool has_avx2() {
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("bmi2") && __builtin_cpu_supports("popcnt");
}
#else
static bool has_avx2() { return false; }
#endif

int main() { return has_avx2() ? 0 : 1; }
