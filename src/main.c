#define _CRT_SECURE_NO_WARNINGS
#include <stdlib.h>
#include <math.h>
#include <conio.h>
#include <stdio.h>
#include <time.h>


#define		subjNum			12		//number of subjects
#define		dimLayer_1		6		//Neural number of the layer 1
#define		dimLayer_2		9		//Neural number of the layer 2
#define		dimLayer_3		2		//Neural number of the layer 3
#define		preWeights		100		//save the weights


double		inLayer_1[dimLayer_1];		//layer 1 inputs data of a single subject
double		inLayer_2[dimLayer_2];		//inputs of layer 2
double		inLayer_3[dimLayer_3];		//inputs of layer 3
double		outTeachVec[dimLayer_3];	//expected output value ( teacher data )
double		neuFiber_1_2[dimLayer_2][dimLayer_1];	//weights from layer 1 to layer 2
double		neuFiber_2_3[dimLayer_3][dimLayer_2];	//weights from layer 2 to layer 3

double		outLayer_2[dimLayer_2];		//outputs of layer 2
double		outLayer_3[dimLayer_3];		//outputs of layer 3
double		biasVec_2[dimLayer_2];		//bias of layer 2
double		biasVec_3[dimLayer_3];		//bias of layer 3
double		subjErr[subjNum];		//the total error of a subject
double		alpha_1;		//learning rate from layer 3 to layer 2
double		alpha_2;		//learning rate from layer 2 to layer 1
double		momentum;		//momentum factor to improve the bpnn algorithm
double		b_errLayer_2[dimLayer_2];		//error of layer 2 output vector 
double		b_errLayer_1[dimLayer_3];		//error of layer 3 output vector


FILE	*fp;
struct {
	double	subjInput[dimLayer_1];
	double	subjTeachOutput[dimLayer_3];
}subj_Data[subjNum];				//container to save the sample

struct {
	double	pre_neuFiber_1_2[dimLayer_2][dimLayer_1];
	double	pre_neuFiber_2_3[dimLayer_3][dimLayer_2];
}pre_WM[preWeights];					//used to save the old weights


int help_window() {
	printf(" 1.Path of the data: 'd:\\bp\\data.txt'!\n");
	printf(" 2.Results will be saved in: 'd:\\bp\\'!\n");
	printf(" 3.The program of BP can study itself for no more than 90000 times.\n And surpassing the number, the program will be ended by itself in\n preventing running infinitely because of error!\n");
	printf("\n\n\n");
	printf("Now press any key to start...\n");

	_getch();
	return 1;
}		// display the start window


int finish_window() { 
	printf("\n\n---------------------------------------------------\n"); 
	printf("This is the end of the program!\n\n"); 
	printf("\n Press any key to exit...\n"); 
	_getch(); 
	exit(0); 
}		// display the ending window


// get the trainning data
int read_subject_data() {
	int m, i, j;
	double input_value;
	if ((fp = fopen("d:\\bp\\data2.txt", "r")) == NULL) {
		printf("Cannot open file and press any key to exit!");
		_getch();
		exit(1);
	}
	m = 0;
	i = 0;
	j = 0;
	while (fscanf(fp, "%lf", &input_value) != EOF) {
		j++;
		if (j <= (subjNum*dimLayer_1)) {
			if (i < dimLayer_1) {
				subj_Data[m].subjInput[i] = input_value;
				printf("\nthe subj_Data[%d].subjInput[%d]=%f\n", m, i, subj_Data[m].subjInput[i]); 
			}
			if (m == (subjNum - 1) && i == (dimLayer_1 - 1)) {
				m = 0;
				i = -1;
			}
			if (i == (dimLayer_1 - 1)) {
				m++;
				i = -1;
			}
		}
		else if ((subjNum*dimLayer_1) < j&&j <= (subjNum*(dimLayer_1 + dimLayer_3))) {
			if (i < dimLayer_3) {
				subj_Data[m].subjTeachOutput[i] = input_value;
				printf("\nThe subj_Data[%d].subjTeachOutput[%d]=%f", m, i, subj_Data[m].subjTeachOutput[i]); 
			}
			if (m == (subjNum - 1) && i == (dimLayer_3 - 1))
				printf("\n");
			if (i == (dimLayer_3 - 1)) {
				m++;
				i = -1;
			}
		}
		i++;
	}
	fclose(fp);
	printf("\nThere are [%d] numbers that have been loaded successfully!\n", j);
	printf("\nShow the data which has been loaded as follows:\n"); 
	for (m = 0; m<subjNum; m++) { 
		for (i = 0; i<dimLayer_1; i++) { 
			printf("\nStudy_Data[%d].subjInput[%d]=%lf", m, i, subj_Data[m].subjInput[i]); 
		} 
		for (j = 0; j<dimLayer_3; j++) { 
			printf("\nStudy_Data[%d].subjTeachOutput[%d]=%lf", m, j, subj_Data[m].subjTeachOutput[j]); 
		} 
	} 
//	_getch();
	printf("\n\nPress any key to start calculating...\n"); 	 
	return 1;
}


int initial_weights() {
	int		i;
	int		ii;
	int		j;
	int		jj;
	int		k;
	int		kk;
	srand((unsigned)time(NULL));
	for (i = 0; i < dimLayer_2; i++) {
		for (j = 0; j < dimLayer_1; j++) {
			neuFiber_1_2[i][j] = (double)((rand() / 32767.0) * 2 - 1);
			printf("neuFiber_1_2[%d][%d]=%f\n", i, j, neuFiber_1_2[i][j]);
		}
	}
	for (ii = 0; ii < dimLayer_3; ii++) {
		for (jj = 0; jj < dimLayer_2; jj++) {
			neuFiber_2_3[ii][jj] = (double)((rand() / 32767.0) * 2 - 1);
			printf("neuFiber_2_3[%d][%d]=%f\n", ii, jj, neuFiber_2_3[ii][jj]);
		}
	}
	for (k = 0; k < dimLayer_2; k++) {
		biasVec_2[k] = (double)((rand() / 32767.0) * 2 - 1);
		printf("biasVec_2[%d]=%f\n", k, biasVec_2[k]);
	}
	for (kk = 0; kk < dimLayer_3; kk++) { 
		biasVec_3[kk] = (double)((rand() / 32767.0) * 2 - 1);
	} 
	return 1;
}


int inNet(int m) {
	int i;
	for (i = 0; i<dimLayer_1; i++) { 
		inLayer_1[i] = subj_Data[m].subjInput[i]; 
	} 
	return 1;
}

int expectedOut(int m) {
	int k;
	for (k = 0; k<dimLayer_3; k++) 
		outTeachVec[k] = subj_Data[m].subjTeachOutput[k]; 
	return 1;
}

int dataFlow1_2() {
	double sigma;
	int i, j;
	for (j = 0; j < dimLayer_2; j++) {
		sigma = 0;
		for (i = 0; i < dimLayer_1; i++) {
			sigma += neuFiber_1_2[j][i] * inLayer_1[i];
		}
		inLayer_2[j] = sigma - biasVec_2[j];
		outLayer_2[j] = 1.0 / (1.0 + exp(-inLayer_2[j]));
	}
	return 1;
}

int dataFlow2_3() {
	int k;
	int j;
	double  sigma;
	for (k = 0; k < dimLayer_3; k++) {
		sigma = 0.0;
		for (j = 0; j < dimLayer_2; j++) {
			sigma += neuFiber_2_3[k][j] * outLayer_2[j];
		}
		inLayer_3[k] = sigma - biasVec_3[k];
		outLayer_3[k] = 1.0 / (1.0 + exp(-inLayer_3[k]));
	}
	return 1;
}

int errFlow3_2(int m) {
	int k;
	double errVec_1[dimLayer_3];
	double errScal_1 = 0;
	for (k = 0; k < dimLayer_3; k++) {
		errVec_1[k] = outTeachVec[k] - outLayer_3[k];
		errScal_1 += (errVec_1[k])*(errVec_1[k]);
		b_errLayer_1[k] = errVec_1[k] * outLayer_3[k] * (1 - outLayer_3[k]);
	}
	subjErr[m] = errScal_1 / 2;
	return 1;
}


int errFlow2_1() {
	int j;
	int	k;
	double sigma;
	for (j = 0; j < dimLayer_2; j++) {
		sigma = 0.0;
		for (k = 0; k < dimLayer_3; k++) {
			sigma += b_errLayer_1[k] * neuFiber_2_3[k][j];
		}
		b_errLayer_2[j] = sigma*outLayer_2[j] * (1 - outLayer_2[j]);
	}
	return 1;
}


int savePreWeights(int m) {
	int i;
	int ii;
	int j;
	int jj;
	for (i = 0; i < dimLayer_2; i++) {
		for (j = 0; j < dimLayer_1; j++) {
			pre_WM[m].pre_neuFiber_1_2[i][j] = neuFiber_1_2[i][j];
		}
	}
	for (ii = 0; ii < dimLayer_3; ii++) {
		for (jj = 0; jj < dimLayer_2; jj++) {
			pre_WM[m].pre_neuFiber_2_3[ii][jj] = neuFiber_2_3[ii][jj];
		}
	}
	return 1;
}


int updateNeuFiber2_3(int n) {
	int k;
	int j;
	if (n < 1) {
		for (k = 0; k < dimLayer_3; k++) {
			for (j = 0; j < dimLayer_2; j++) {
				neuFiber_2_3[k][j] = neuFiber_2_3[k][j] + alpha_1*b_errLayer_1[k] * outLayer_2[j];
//				printf("neuFiber_2_3[%d][%d] = %lf\n", k, j, neuFiber_2_3[k][j]);
			}
			biasVec_3[k] += alpha_1*b_errLayer_1[k];
//			printf("biasVec_3[%d] = %lf\n", k, biasVec_3[k]);
		}
	}
	else if (n > 1) {
		for (k = 0; k < dimLayer_3; k++) {
			for (j = 0; j < dimLayer_2; j ++ ) {
				neuFiber_2_3[k][j] = neuFiber_2_3[k][j] + alpha_1*b_errLayer_1[k] * outLayer_2[j] + momentum*(neuFiber_2_3[k][j] - pre_WM[(n - 1)].pre_neuFiber_2_3[k][j]);
//				printf("neuFiber_2_3[%d][%d] = %lf\n", k,j, neuFiber_2_3[k][j]);
			}
			biasVec_3[k] += alpha_1*b_errLayer_1[k];
//			printf("biasVec_3[%d] = %lf\n", k, biasVec_3[k]);
		}
	}
	return 1;
}


int updateNeuFiber1_2(int n) {
	int i;
	int j;
	if (n <= 1) {
		for (j = 0; j < dimLayer_2; j++) {
			for (i = 0; i < dimLayer_1; i++) {
				neuFiber_1_2[j][i] = neuFiber_1_2[j][i] + alpha_2*b_errLayer_2[j] * inLayer_1[i];
//				printf("neuFiber_1_2[%d][%d] = %lf\n", j, i, neuFiber_1_2[j][i]);
			}
			biasVec_2[j] += alpha_2*b_errLayer_2[j];
//			printf("biasVec_2[%d] = %lf\n", j, biasVec_2[j]);
		}
	}
	else if (n > 1) {
		for (j = 0; j < dimLayer_2; j++) {
			for (i = 0; i < dimLayer_1; i++) {
				neuFiber_1_2[j][i] = neuFiber_1_2[j][i] + alpha_2*b_errLayer_2[j] * inLayer_1[i] + momentum*(neuFiber_1_2[j][i] - pre_WM[(n - 1)].pre_neuFiber_1_2[j][i]);
//				printf("neuFiber_1_2[%d][%d] = %lf\n", j, i, neuFiber_1_2[j][i]);
			}
			biasVec_2[j] += alpha_2*b_errLayer_2[j];
//			printf("biasVec_2[%d] = %lf\n", j, biasVec_2[j]);
		}
	}
	return 1;
}


double calculate_total_error() {
	int m;
	double total_err = 0;
	for (m = 0; m < subjNum; m++) {
		total_err += subjErr[m];
	}
	return total_err;
}


void saveWeight() {
	int i;
	int j;
	int k;
	int ii;
	int jj;
	int kk;
	if ((fp = fopen("d:\\bp\\weight.txt", "a")) == NULL) {
		printf("Cannot open file strike any key exit!");
		_getch();
		exit(1);
	}
	fprintf(fp, "Save the result of 'weight' as follows:\n");
	for (i = 0; i < dimLayer_2; i++) {
		for (j = 0; j < dimLayer_1; j++) {
			fprintf(fp, "neuFiber_1_2[%d][%d]=%f\n", i, j, neuFiber_1_2[i][j]);
		}
	}
	fprintf(fp, "\n");
	for (ii = 0; ii < dimLayer_3; ii++) {
		for (jj = 0; jj < dimLayer_2; jj++) {
			fprintf(fp, "neuFiber_2_3[%d][%d]=%f\n", ii, jj, neuFiber_2_3[ii][jj]);
		}
	}
	fclose(fp);
	printf("\nThe result of 'weight.txt' has been saved successfully!\nPress any key to continue...");

	if ((fp = fopen("d:\\bp\\b.txt", "a")) == NULL) {
		printf("Cannot open file strike any key exit!");
		_getch();
		exit(1);
	}
	fprintf(fp, "Save the result of 'b value' as follows:\n");
	for (k = 0; k<dimLayer_3; k++) 
		fprintf(fp, "biasVec_3[%d]=%f\n", k, biasVec_3[k]);
	fprintf(fp, "\nSave the result of 'b value' as follows:\n");
	for (kk = 0; kk < dimLayer_2; kk++)
		fprintf(fp, "biasVec_2[%d]=%f\n", kk, biasVec_2[kk]);
	fclose(fp);
	printf("\nThe result of 'b.txt' has been save successfully\nPress any key to continue...");
	_getch();
}


void main() {
	double targetErr;
	double totalErr;
	long iterationCount;
	int flag;
	flag = 90000;

	iterationCount = 0;
	targetErr = 0.01;
	help_window();
	read_subject_data();
	initial_weights();


//	srand(time(NULL));
	momentum = 0.9;//(double)((rand() / 16383.0) - 0.8);
	alpha_1 = 0.7;//(double)((rand() / 16383.0) - 0.8);// (double)((rand() / 20.0) / 100);
	alpha_2 = 0.7;//(double)((rand() / 16383.0) - 0.8);// (double)((rand() / 20.0) / 100);


	do {
		int m;
		++iterationCount;
		for (m = 0; m < subjNum; m++) {
			inNet(m);
			expectedOut(m);
			dataFlow1_2();
			dataFlow2_3();
			errFlow3_2(m);
			errFlow2_1();
			savePreWeights(m);
			updateNeuFiber2_3(m);
			updateNeuFiber1_2(m);
		}
		totalErr = calculate_total_error();
		printf("totalErr=%f\n", totalErr);
		printf("targetErr=%f\n\n", targetErr);
		if (iterationCount>flag) { 
			printf("\n*******************************\n"); 
			printf("The program is ended by itself!\n The learning times is surpassed!\n"); 
			printf("*****************************\n"); 
			_getch(); 
			break; 
		}
	}while (totalErr > targetErr);
	printf("\n****************\n"); 
	printf("\nThe program have studied for [%ld] times!\n", iterationCount); 
	printf("\n****************\n"); 
	saveWeight(); 
	finish_window();
}