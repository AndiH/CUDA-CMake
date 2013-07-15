#include <iostream>
#include "externalClass.cu" // important to include .cu file, not header file

int main() {
	externalClass myStuff;

	std::cout << "This is just a plain, host-generated 5, isn't it?: " << myStuff.GetInt() << std::endl;

	const int localN = 10;
	double * localFloat = new double[localN];
	for (int i = 0; i < localN; i++)
		localFloat[i] = i;

	myStuff.squareOnDevice(localFloat, localN);
	
	std::cout << "Final squared values are: " << std::endl;
	
	for (int i = 0; i < localN; i++)
		std::cout << localFloat[i] << std::endl;

	return 1;
}
