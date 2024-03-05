#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>

#define NUM_STUDENTS 4

struct student {
	char *forename;
	char *surname;
	float average_module_mark;

};

void print_student(const struct student *student);

void main() {
	struct student *students = (struct student*)malloc(NUM_STUDENTS * sizeof(struct student));
	int i;
	unsigned int forename_length,surname_length;

	FILE* f = NULL;
	f = fopen("students2.bin", "rb"); //read and binary flags
	if (f == NULL) {
		fprintf(stderr, "Error: Could not find students2.bin file \n");
		exit(1);
	}

	//fread(students, sizeof(struct student), NUM_STUDENTS, f);

	for (i = 0; i < NUM_STUDENTS; i++) {
		fread(&forename_length,sizeof(unsigned int), 1, f);
		students[i].forename = (char*)malloc((forename_length + 1) * sizeof(char));
		fread(students[i].forename, sizeof(char), forename_length, f); // read forename
		students[i].forename[forename_length] = '\0';

		fread(&surname_length, sizeof(unsigned int), 1, f);
		students[i].surname = (char*)malloc((surname_length + 1) * sizeof(char));
		fread(students[i].surname, sizeof(char), surname_length, f); // read surname
		students[i].surname[surname_length] = '\0';

		fread(&students[i].average_module_mark, sizeof(float), 1, f);

		print_student(&students[i]);
		free(students[i].forename);
		free(students[i].surname);
	}
	fclose(f);
	free(students);
}

void print_student(const struct student *student) {
	printf("Student:\n");
	printf("\tForename: %s\n", student->forename);
	printf("\tSurname: %s\n", student->surname);
	printf("\tAverage Module Mark: %.2f\n", student->average_module_mark);
}

