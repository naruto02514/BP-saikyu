#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#define INPUT 2
#define OUTPUT 1
#define HIDDEN 2

double OT_IN[INPUT+1]; //���͑w�̏o��-�f�[�^�Z�b�g����
double OT_HN[HIDDEN+1]; //���ԑw�̏o��(Hj)
double OT_OT[OUTPUT]; //�o�͑w�̏o��(Ok)
double W_IN[(INPUT+1)*HIDDEN]; //�����W��(Wkj)
double W_HN[(HIDDEN+1)*OUTPUT]; //�����W��(Vkj)
double TEACH[OUTPUT]; //���t�M��
double DEL_OT[OUTPUT]; //�덷(��k)
double DEL_HN[HIDDEN]; //�덷(�Ђ�)

/*�ꊇ�C���֐�*/
double DW_IN[(INPUT + 1)*HIDDEN];
double DW_HN[(HIDDEN + 1)*OUTPUT];

double alpha; //�萔��
double erlimit; //�덷���e�̏���l
double u0; //�V�O���C�h�̌X��
double ru0;
double intwgt; //�����W���̏����l
double intoff; //�I�t�Z�b�g�̏����l
double error; //�S�̂̌덷
double tstart, tstop, ttime;
double atime = 0;

int times; //�w�K��
int ok=0;
int nook=0;
long int sumloop=0;
int gaku;

/*���t�M���̃f�[�^�Z�b�g*/
/*------------------*/
struct {
	double input[2];
	double tch[1];
} indata[4]={{1.0,0.0,0.99000},{0.0,1.0,0.99000},{0.0,0.0,0.01000},{1.0,1,0.01000},};

/*�����쐬*/
/*--------*/
double drand48(){
	double r;
	r=(double)rand()/32767;
	return r;
}

/*�����쐬*/
/*--------*/
double srand48(){
	double s=0.0;
	while (s < 0.10 || s > 0.99) s=(double)rand()/32767;
	return s;
}
	
	
/*�V�O���C�h�֐�*/
/*-----------*/
double sigmoid(double u){
	double s;
	s = u/u0;
	s = 0.5*(1.0+tanh(s)); 
	if(s>0.99000) return(0.99000);
	else if(s<0.01000) return(0.01000);
	else return(s);
}

/*�f�[�^����*/
/*----------*/
void scan_data(FILE *fp){
	printf("�V�O���C�h�̌X��(u0)=");
	scanf_s("%lf",&u0);
	fprintf(fp, "�V�O���C�h�̌X��(u0)= %lf\n",u0);
	ru0 = 2.0 / u0;
	printf("�w�K�p�����^(�A���t�@)=");
	scanf_s("%lf",&alpha);
	fprintf(fp, "�w�K�p�����^(�A���t�@)= %lf\n",alpha);
	printf("�덷���e�̏���l=");
	scanf_s("%lf",&erlimit);
	fprintf(fp, "�덷���e�̏���l= %lf\n",erlimit);
	printf("�w�K�񐔂̏���l=");
	scanf_s("%d",&times);
	fprintf(fp, "�w�K�񐔂̏���l= %d\n",times);

}

/*�f�[�^�𗐐��ɏ�����*/
/*------------------*/
void initial_data(FILE *fp){
	int a,b;
	int count;
	count = 0;
	for(a=0;a<HIDDEN;a++){
		for(b=0;b<INPUT;b++){
			W_IN[count]= intwgt*(drand48()-0.5)*2.0;
			fprintf(fp, "W_IN[%d]=%f\n", count, W_IN[count]);
			count++;
		}
		W_IN[count] = intoff*drand48();
		fprintf(fp, "W_IN[%d]=%f\n", count, W_IN[count]);
		count++;
	}
	count = 0;
	for(a=0;a<OUTPUT;a++){
		for(b=0;b<HIDDEN;b++){
			W_HN[count]= intwgt*(drand48()-0.5)*2.0;
			fprintf(fp, "W_HN[%d]=%f\n", count, W_HN[count]);
			count++;
		}
		W_HN[count] = intoff*drand48();
		fprintf(fp, "W_HN[%d]=%f\n", count, W_HN[count]);
		count++;
	}
}

/*�ꊇ�C���ʃN���A(�ǉ�)*/
void w_clear(){
	int a,b;
	int count;
	count = 0;
	for(a=0;a<HIDDEN;a++){
		for (b = 0; b < INPUT + 1; b++){
			DW_IN[count] = 0.0;
			count++;
		}
	}
	count = 0;
	for(a=0;a<OUTPUT;a++){
		for (b = 0; b < HIDDEN + 1; b++) {
			DW_HN[count] = 0.0;
			count++;
		}
	}
}
	
	
/*�o�b�N�v���o�Q�[�V�����֐�*/
/*------------------*/
void backpropagation(FILE *fp){
	int loopf=0,loop,a,b;
	int count;
	double sum,wkb;
	tstart = (double)clock();
	while(loopf < times){
		loopf++;
		if(loopf % 50 == 0) fprintf(fp,"\n�w�K��%d��\n",loopf);
		error=0.0;
		w_clear();
		for(loop=0;loop<4;loop++){
			sum=0.0;
			if(loopf % 50 == 0) fprintf(fp,"Input  "); 
			/*���͑w�̏o�͂��Z�b�g����*/
			for(a=0;a<INPUT;a++){ 
				OT_IN[a] = (double)indata[loop].input[a];
				if(loopf % 50 == 0) fprintf(fp,"%1.1f ",OT_IN[a]);
			}
			OT_IN[INPUT] = 1.0;
			OT_HN[HIDDEN] = 1.0;
			/*���ԑw�̏o�͂����߂�*/
			count = 0;
			for(a=0;a<HIDDEN;a++){
				for (b = 0, sum = 0.0; b < INPUT+1; b++){
					sum += (W_IN[count] * OT_IN[b]);
					count++;
				}
				OT_HN[a]=sigmoid(sum); //Hj=f(Uj)
			}
			/*�o�͑w�̏o�͂����߂�*/
			count = 0;
			for(a=0;a<OUTPUT;a++){
				for (b = 0,sum = 0.0; b < HIDDEN+1; b++) {
					sum += (W_HN[count] * OT_HN[b]);
					count++;
				}
				OT_OT[a] = sigmoid(sum);//Ok=f(Sk)
				if(loopf % 50 == 0) fprintf(fp,"\nOutput  %lf\n",OT_OT[a]);
			}
			/*�덷�̌v�Z*/
			for(a=0;a<OUTPUT;a++){
				TEACH[a]=(double)indata[loop].tch[a];
				wkb = TEACH[a] - OT_OT[a];
				error += (fabs(wkb));
				/*��k=(Tk-Ok)*Ok*(1-Ok)*/
				DEL_OT[a] = wkb * ru0 * OT_OT[a] * (1.0 - OT_OT[a]);
			}
			/*�덷�̌v�Z*/
			count = 0;
			for(a=0;a<HIDDEN;a++){
				for (b = 0,sum = 0.0; b < OUTPUT; b++){
					sum += (DEL_OT[b] * W_HN[count]);
					count++;
				}
				/*��j=��(k)��k*Vkj*Hj*(1-Hj)*/
				DEL_HN[a] = sum * ru0 * OT_HN[a] * ( 1.0 - OT_HN[a]);
			}
			/*���ԑw����̌����׏d(�C��)*/
			count = 0;
			for(a=0;a<OUTPUT;a++){
				for(b=0;b<HIDDEN+1;b++){
					/*Vkj=Vkj+��*��k*Hj*/
					DW_HN[count] += (alpha * DEL_OT[a] * OT_HN[b]);
					count++;
				}
			}
			/*���͑w����̌����׏d(�C��)*/
			count = 0;
			for(a=0;a<HIDDEN;a++){
				for(b=0;b<INPUT+1;b++){
					/*Wji=Wji+��*��j*Hj*/
					DW_IN[count] += (alpha * DEL_HN[a] * OT_IN[b]);
					count++;
				}
			}
		}
		/*�ꊇ�X�V(�ǉ�)*/
		count = 0;
		for(a=0;a<HIDDEN;a++){
			for (b = 0; b < INPUT + 1; b++) {
				W_IN[count] += DW_IN[count];
				count++;
			}
		}
		count = 0;
		for(a=0;a<OUTPUT;a++){
			for (b = 0; b < HIDDEN+1; b++) {
				W_HN[count] += DW_HN[count];
				count++;
			}
		}	
		
		if (loopf % 50 == 0) fprintf(fp,"-----------\nError  %lf\n",error);
		if (error < erlimit) {
			ok += 1;
			sumloop += loopf;
			tstop = (double)clock();
			ttime = tstop - tstart;
			atime += ttime;
			break;
		}
		if (loopf == times && error > erlimit) nook += 1;
	}
	fprintf(fp,"\n-----------------\n");
	fprintf(fp,"�S�̂̊w�K��%d\n",loopf);
	
}

void print_w(FILE *fp){
	int a, b, count;
	count = 0;
	for(a=0;a<HIDDEN;a++){
		fprintf(fp,"�����W��W[%d] = {",a);
		for(b=0;b<INPUT;b++){
			if(b!=0) fprintf(fp, ",");
			fprintf(fp,"%.6lf",W_IN[count]);
			count++;
		}
		fprintf(fp, "}  �V�[�^ = %.6lf\n", W_IN[count]);
		count++;
	}
	count = 0;
	for(a=0;a<OUTPUT;a++){
		fprintf(fp,"�����W��V[%d] = {",a);
		for(b=0;b<HIDDEN;b++){
			if(b!=0) fprintf(fp, ",");
			fprintf(fp,"%.6lf",W_HN[count]);
			count++;
		}
		fprintf(fp, "}  �K���} = %.6lf\n", W_HN[count]);
		count++;
	}
}

int main(void){
	long double b;
	FILE *fp;
	fopen_s(&fp,"out.txt","w+");
	scan_data(fp);
	for(gaku=0;gaku<10;gaku++){
		srand(gaku); 
		intwgt=srand48();
		intoff=srand48();
		fprintf(fp, "�����W���̏����l[-1.0,1.0] * ? = %lf\n",intwgt);
		fprintf(fp, "�I�t�Z�b�g�̏����l[0.0,1.0] * +-? = %lf\n",intoff);
		fprintf(fp,"\n----�f�[�^�Z�b�g----\n");
		fprintf(fp, "1 1->(0.99000)\n0 1->(0.99000)\n0 0->(0.01000)\n1 1->(0.01000)\n");
		initial_data(fp);
		backpropagation(fp);
		print_w(fp);
		fprintf(fp, "����=%d,���s=%d\n",ok,nook);
		b=(double)sumloop/(double)ok;
		fprintf(fp, "���ϔ�����=%f\n",b);
	}
	atime = atime / (double)ok;
	fprintf(fp, "���ώ���=%lf", atime);
	fclose(fp);
	return 0;
}
			


