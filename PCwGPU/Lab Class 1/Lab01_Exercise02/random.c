#include <stdlib.h>
#include "random.h"
#include <stdint.h>

unsigned int rseed;

void init_random() {
	srand(RAND_SEED);
	rseed = RAND_SEED;
}
unsigned short random_ushort() {
	// Ex 1.7, C-style casts take the form `(<type>)`, this is the same as Java
	return (unsigned short)(rand());
}

unsigned int random_uint() {
	rseed = RANDOM_A*rseed + RANDOM_C;
}

float random_float() {
	rseed = (float)rseed;
	return rseed;
}
