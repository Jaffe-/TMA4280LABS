#include <stdbool.h>
#include <stdio.h>

#define TEST_CASE(expr) \
    printf("* test of %s:\t%s\n", #expr, (expr) ? "PASS" : "FAIL")

#ifdef DO_UNITTEST
#define UNITTEST(func) \
    func();            \
    return 0;
#else
#define UNITTEST(func) do { } while(0);
#endif

#ifdef DO_VTEST
#define VERIFICATION_TEST(func) \
    func();                     \
    return 0;
#else
#define VERIFICATION_TEST(func) do { } while(0);
#endif
