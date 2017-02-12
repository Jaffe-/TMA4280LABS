#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../unittest.h"

double zeta0(int n)
{
    double sum = 0;

    for (int i = 1; i <= n; i++) {
        sum += pow(i, -2);
    }

    return sqrt(6 * sum);
}

void unittests(void)
{
    TEST_CASE(fabs(zeta0(3) - 2.857738) < 0.0001);
}

void verification_test(void)
{
    printf("Comparing to PI = %.17g\n\n", M_PI);
    for (int i = 1; i <= 24; i++) {
        double error = fabs(M_PI - zeta0(pow(2, i)));
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

    printf("%f\n", zeta0(n));

    return 0;
}
