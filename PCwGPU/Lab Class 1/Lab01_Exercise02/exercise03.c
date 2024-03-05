#include <stdio.h>
#include <stdint.h>
#include "random.h"

#define NUM_VALUES 250
signed long long values[NUM_VALUES];

int main()
{
	unsigned long long int sum = 0;
	unsigned char i = 0;

	init_random();


	for (i = 0; i < NUM_VALUES; i++) {
		random_uint();
		values[i] = random_float();
		printf("i: %u, value: %0.0f\n", i, values[i]); // Debug statement
		sum += values[i];
	}
	printf("sum: %0.0f", sum);
	unsigned int average;
	average = (float)(sum / NUM_VALUES);
	signed long long min = 0;
	signed long long max = 0;

	for (i = 0; i < NUM_VALUES; i++) {
		values[i] -= average;
		min = (values[i] < min) ? values[i] : min;
		max = (values[i] > max) ? values[i] : max;
	}
	printf("\nAverage: %0.0f\n", average);
	printf("Min: %0.0f\n", (float)min);
	printf("Max: %0.0f\n", (float)max);

	return 0;
}