#include <iostream>

#include "externalClass.h"

int main() {

	const int N = 10;

	double * data = new double[N];
	for (int i = 0; i < N; ++i)
		data[i] = i;
	
	externalClass example;

	example.cube(data, N);

	std::cout << "Cubed values are: " << std::endl;
	for (int i = 0; i < N; ++i) 
		std::cout << data[i] << std::endl;

}
