#define _CRT_SECURE_NO_WARNINGS
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include "mandelbrot.h"

// Image size
#define WIDTH 1024
#define HEIGHT 768

#define MAX_ITERATIONS 15				// Number of iterations

// C parameters (modify these to change the zoom and position of the Mandelbrot)
#define ZOOM 1.0
#define X_DISPLACEMENT -0.5
#define Y_DISPLACEMENT 0.0

static int iterations[HEIGHT][WIDTH];					// Store the escape time (iteration count) as an integer
static double iterations_d[HEIGHT][WIDTH];				// Store for the escape time as a double (with fractional part) for NIC method only
static int histogram[MAX_ITERATIONS + 1];				// Histogram of escape times
static rgb rgb_output[HEIGHT][WIDTH];	// Output data
rgb h_ev_transfer(int x, int y);


const TRANSFER_FUNCTION tf = ESCAPE_VELOCITY;

int main(int argc, char* argv[]) {
    int x, y;											// Loop counters
    double c_r, c_i;									// Real and imaginary part of the constant c
    double n_r, n_i, o_r, o_i;							// Real and imaginary parts of new and old z
    double mu;											// Iteration with fractional component
    double begin, end;									// Timers
    double elapsed;										// Elapsed time
    FILE* f;											// Output file handle

    // Open the output file and write header info for PPM filetype
    f = fopen("output.ppm", "wb");
    if (f == NULL) {
        fprintf(stderr, "Error opening 'output.ppm' output file\n");
        exit(1);
    }
    fprintf(f, "P6\n");
    fprintf(f, "# COM4521 Lab 03 Exercise02\n");
    fprintf(f, "%d %d\n%d\n", WIDTH, HEIGHT, 255);

    // Start timer
    begin = omp_get_wtime();

    // Clear the histogram initial values
    memset(histogram, 0, sizeof(histogram));

    // STAGE 1) Calculate the escape time for each pixel
#pragma omp parallel for default(none) private(x, y, n_r, n_i, o_r, o_i, c_r, c_i, mu) shared(iterations, iterations_d, histogram, tf)
    for (y = 0; y < HEIGHT; y++) {
        for (x = 0; x < WIDTH; x++) {
            // Initialize complex number values
            n_r = n_i = o_r = o_i = 0.0;

            // Calculate the initial real and imaginary part of z
            c_r = 1.5 * (x - WIDTH / 2) / (0.5 * ZOOM * WIDTH) + X_DISPLACEMENT;
            c_i = (y - HEIGHT / 2) / (0.5 * ZOOM * HEIGHT) + Y_DISPLACEMENT;

            int i;
            for (i = 0; (i < MAX_ITERATIONS) && ((n_r * n_r + n_i * n_i) < ESCAPE_RADIUS_SQ); i++) {
                // Store current values
                o_r = n_r;
                o_i = n_i;

                // Apply Mandelbrot function
                n_r = o_r * o_r - o_i * o_i + c_r;
                n_i = 2.0 * o_r * o_i + c_i;
            }

            // Escape time algorithm for certain transfer functions
            if (tf == HISTOGRAM_NORMALISED_ITERATION_COUNT && i < MAX_ITERATIONS) {
                mu = i - log(log(sqrt(n_r * n_r + n_i * n_i))) / log(2);
                iterations_d[y][x] = mu;
                i = (int)mu;
            }

            iterations[y][x] = i; // Record the escape velocity

            if (tf == HISTOGRAM_ESCAPE_VELOCITY || tf == HISTOGRAM_NORMALISED_ITERATION_COUNT) {
#pragma omp atomic update
                histogram[i]++;
            }
        }
    }

    // STAGE 2) Calculate the transfer (rgb output) for each pixel
    for (y = 0; y < HEIGHT; y++) {
        for (x = 0; x < WIDTH; x++) {
            switch (tf) {
            case ESCAPE_VELOCITY:
                rgb_output[y][x] = ev_transfer(x, y);
                break;
            case HISTOGRAM_ESCAPE_VELOCITY:
                rgb_output[y][x] = h_ev_transfer(x, y);
                break;
            case HISTOGRAM_NORMALISED_ITERATION_COUNT:
                rgb_output[y][x] = h_nic_transfer(x, y);
                break;
            }
        }
    }

    // STAGE 3) Output the Mandelbrot to a file
    fwrite(&rgb_output, sizeof(rgb), HEIGHT * WIDTH, f);
    fclose(f);

    // Stop timer
    end = omp_get_wtime();
    elapsed = end - begin;
    printf("Complete in %f seconds\n", elapsed);

    return 0;
}

// Transfer function implementations
rgb ev_transfer(int x, int y) {
    rgb a;
    double hue;
    int its = iterations[y][x];
    if (its == MAX_ITERATIONS) {
        a.r = a.g = a.b = 0;
    }
    else {
        hue = its / (double)MAX_ITERATIONS;
        a.r = a.g = 0;
        a.b = (char)(hue * 255.0); // Clamp to range of 0-255
    }
    return a;
}


rgb h_ev_transfer(int x, int y) {
    rgb a;
    double hue;
    int its;
    int i;

    its = iterations[y][x];
    if (its == MAX_ITERATIONS) {
        a.r = a.g = a.b = 0;
    }
    else {
        hue = 0;
        for (i = 0; i < its; i++) {
            hue += (histogram[i] / (double)(WIDTH * HEIGHT));
        }
        a.r = a.g = 0;
        a.b = (unsigned char)(hue * 255.0); // Clamp to range of 0-255
    }
    return a;
}



rgb h_nic_transfer(int x, int y) {
    rgb a;
    double hue1, hue2, its_d, frac;
    int its = iterations[y][x];
    its_d = iterations_d[y][x];

    hue1 = hue2 = 0;
    for (int i = 0; i < its && its < MAX_ITERATIONS; i++)
        hue1 += (histogram[i] / (double)(HEIGHT * WIDTH));
    if (its <= MAX_ITERATIONS)
        hue2 = hue1 + (histogram[its] / (double)(HEIGHT * WIDTH));

    a.r = a.g = 0;
    frac = its_d - (int)its_d;
    double hue = (1 - frac) * hue1 + frac * hue2; // Linear interpolation between hues
    a.b = (char)(hue * 255.0);                    // Clamp to range of 0-255

    return a;
}
