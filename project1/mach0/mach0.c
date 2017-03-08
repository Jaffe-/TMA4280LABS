#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../unittest.h"

double comp_arctan(double x, int n)
{
    double sum = 0;

    for (long int i = 0; i <= n; i++) {
        sum += ((i % 2) ? (-1) : 1) * pow(x, 2 * i + 1) / (2 * i + 1);
    }

    return sum;
}

double mach0(int n)
{
    return 16 * comp_arctan(1.0/5, n) - 4 * comp_arctan(1.0/239, n);
}

void unittests()
{
    TEST_CASE(fabs(mach0(3) - 3.141592) < 0.001);
}

void verification_test(void)
{
    printf("Comparing to PI = %.17g\n\n", M_PI);
    for (int i = 1; i <= 24; i++) {
        double error = fabs(M_PI - mach0(pow(2, i)));
        printf("n = %d: %.17g\n", (int)pow(2, i), error);
    }
}

int main(int argc, char **argv)
{
    UNITTEST(unittests);
    VERIFICATION_TEST(verification_test);

    if (argc != 2) {
        printf("Exactly one argument expected\n");
        return -1;
    }

    int n = atoi(argv[1]);

    printf("%f\n", mach0(n));

    return 0;
}
