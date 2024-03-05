#include <stdio.h>
#include "random.h"
#include <stdint.h>

#define NUM_VALUES 250
int32_t values[NUM_VALUES];

int main()
{
	uint32_t sum = 0;
	uint8_t i = 0;

	init_random();

	for (i = 0; i < NUM_VALUES; i++) {
		values[i] = (int)random_ushort();
		printf("i: %d, value: %d\n", i, values[i]); // Debug statement
		sum += values[i];
	}
	printf("sum: %d", sum);
	float average = 0.0;
	average = sum / NUM_VALUES;
	int min = INT_MAX;
	int max = INT_MIN;


	for (i = 0; i < NUM_VALUES; i++) {
		values[i] -= average;
		min = (values[i] < min) ? values[i] : min;
		max = (values[i] > max) ? values[i] : max;
	}
	printf("\nAverage: %0.1f\n", average);
	printf("Min: %d\n", min);
	printf("Max: %d\n", max);
	
	return 0;
}