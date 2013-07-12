#include "externalClass.h"


void externalClass::cube(double *a, const int N) {
	for (int i = 0; i < N; i++) 
		a[i] = a[i] * a[i] * a[i];
}
