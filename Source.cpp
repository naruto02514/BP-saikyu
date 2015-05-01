#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#define INPUT 2
#define OUTPUT 1
#define HIDDEN 2

double OT_IN[INPUT+1]; //入力層の出力-データセット入り
double OT_HN[HIDDEN+1]; //中間層の出力(Hj)
double OT_OT[OUTPUT]; //出力層の出力(Ok)
double W_IN[(INPUT+1)*HIDDEN]; //結合係数(Wkj)
double W_HN[(HIDDEN+1)*OUTPUT]; //結合係数(Vkj)
double TEACH[OUTPUT]; //教師信号
double DEL_OT[OUTPUT]; //誤差(δk)
double DEL_HN[HIDDEN]; //誤差(σｊ)

/*一括修正関数*/
double DW_IN[(INPUT + 1)*HIDDEN];
double DW_HN[(HIDDEN + 1)*OUTPUT];

double alpha; //定数α
double erlimit; //誤差許容の上限値
double u0; //シグモイドの傾き
double ru0;
double intwgt; //結合係数の初期値
double intoff; //オフセットの初期値
double error; //全体の誤差
double tstart, tstop, ttime;
double atime = 0;

int times; //学習回数
int ok=0;
int nook=0;
long int sumloop=0;
int gaku;

/*教師信号のデータセット*/
/*------------------*/
struct {
	double input[2];
	double tch[1];
} indata[4]={{1.0,0.0,0.99000},{0.0,1.0,0.99000},{0.0,0.0,0.01000},{1.0,1,0.01000},};

/*乱数作成*/
/*--------*/
double drand48(){
	double r;
	r=(double)rand()/32767;
	return r;
}

/*乱数作成*/
/*--------*/
double srand48(){
	double s=0.0;
	while (s < 0.10 || s > 0.99) s=(double)rand()/32767;
	return s;
}
	
	
/*シグモイド関数*/
/*-----------*/
double sigmoid(double u){
	double s;
	s = u/u0;
	s = 0.5*(1.0+tanh(s)); 
	if(s>0.99000) return(0.99000);
	else if(s<0.01000) return(0.01000);
	else return(s);
}

/*データ入力*/
/*----------*/
void scan_data(FILE *fp){
	printf("シグモイドの傾き(u0)=");
	scanf_s("%lf",&u0);
	fprintf(fp, "シグモイドの傾き(u0)= %lf\n",u0);
	ru0 = 2.0 / u0;
	printf("学習パラメタ(アルファ)=");
	scanf_s("%lf",&alpha);
	fprintf(fp, "学習パラメタ(アルファ)= %lf\n",alpha);
	printf("誤差許容の上限値=");
	scanf_s("%lf",&erlimit);
	fprintf(fp, "誤差許容の上限値= %lf\n",erlimit);
	printf("学習回数の上限値=");
	scanf_s("%d",&times);
	fprintf(fp, "学習回数の上限値= %d\n",times);

}

/*データを乱数に初期化*/
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

/*一括修正量クリア(追加)*/
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
	
	
/*バックプロバゲーション関数*/
/*------------------*/
void backpropagation(FILE *fp){
	int loopf=0,loop,a,b;
	int count;
	double sum,wkb;
	tstart = (double)clock();
	while(loopf < times){
		loopf++;
		if(loopf % 50 == 0) fprintf(fp,"\n学習第%d回\n",loopf);
		error=0.0;
		w_clear();
		for(loop=0;loop<4;loop++){
			sum=0.0;
			if(loopf % 50 == 0) fprintf(fp,"Input  "); 
			/*入力層の出力をセットする*/
			for(a=0;a<INPUT;a++){ 
				OT_IN[a] = (double)indata[loop].input[a];
				if(loopf % 50 == 0) fprintf(fp,"%1.1f ",OT_IN[a]);
			}
			OT_IN[INPUT] = 1.0;
			OT_HN[HIDDEN] = 1.0;
			/*中間層の出力を求める*/
			count = 0;
			for(a=0;a<HIDDEN;a++){
				for (b = 0, sum = 0.0; b < INPUT+1; b++){
					sum += (W_IN[count] * OT_IN[b]);
					count++;
				}
				OT_HN[a]=sigmoid(sum); //Hj=f(Uj)
			}
			/*出力層の出力を求める*/
			count = 0;
			for(a=0;a<OUTPUT;a++){
				for (b = 0,sum = 0.0; b < HIDDEN+1; b++) {
					sum += (W_HN[count] * OT_HN[b]);
					count++;
				}
				OT_OT[a] = sigmoid(sum);//Ok=f(Sk)
				if(loopf % 50 == 0) fprintf(fp,"\nOutput  %lf\n",OT_OT[a]);
			}
			/*誤差の計算*/
			for(a=0;a<OUTPUT;a++){
				TEACH[a]=(double)indata[loop].tch[a];
				wkb = TEACH[a] - OT_OT[a];
				error += (fabs(wkb));
				/*δk=(Tk-Ok)*Ok*(1-Ok)*/
				DEL_OT[a] = wkb * ru0 * OT_OT[a] * (1.0 - OT_OT[a]);
			}
			/*誤差の計算*/
			count = 0;
			for(a=0;a<HIDDEN;a++){
				for (b = 0,sum = 0.0; b < OUTPUT; b++){
					sum += (DEL_OT[b] * W_HN[count]);
					count++;
				}
				/*σj=Σ(k)δk*Vkj*Hj*(1-Hj)*/
				DEL_HN[a] = sum * ru0 * OT_HN[a] * ( 1.0 - OT_HN[a]);
			}
			/*中間層からの結合荷重(修正)*/
			count = 0;
			for(a=0;a<OUTPUT;a++){
				for(b=0;b<HIDDEN+1;b++){
					/*Vkj=Vkj+α*δk*Hj*/
					DW_HN[count] += (alpha * DEL_OT[a] * OT_HN[b]);
					count++;
				}
			}
			/*入力層からの結合荷重(修正)*/
			count = 0;
			for(a=0;a<HIDDEN;a++){
				for(b=0;b<INPUT+1;b++){
					/*Wji=Wji+α*σj*Hj*/
					DW_IN[count] += (alpha * DEL_HN[a] * OT_IN[b]);
					count++;
				}
			}
		}
		/*一括更新(追加)*/
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
	fprintf(fp,"全体の学習回数%d\n",loopf);
	
}

void print_w(FILE *fp){
	int a, b, count;
	count = 0;
	for(a=0;a<HIDDEN;a++){
		fprintf(fp,"結合係数W[%d] = {",a);
		for(b=0;b<INPUT;b++){
			if(b!=0) fprintf(fp, ",");
			fprintf(fp,"%.6lf",W_IN[count]);
			count++;
		}
		fprintf(fp, "}  シータ = %.6lf\n", W_IN[count]);
		count++;
	}
	count = 0;
	for(a=0;a<OUTPUT;a++){
		fprintf(fp,"結合係数V[%d] = {",a);
		for(b=0;b<HIDDEN;b++){
			if(b!=0) fprintf(fp, ",");
			fprintf(fp,"%.6lf",W_HN[count]);
			count++;
		}
		fprintf(fp, "}  ガンマ = %.6lf\n", W_HN[count]);
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
		fprintf(fp, "結合係数の初期値[-1.0,1.0] * ? = %lf\n",intwgt);
		fprintf(fp, "オフセットの初期値[0.0,1.0] * +-? = %lf\n",intoff);
		fprintf(fp,"\n----データセット----\n");
		fprintf(fp, "1 1->(0.99000)\n0 1->(0.99000)\n0 0->(0.01000)\n1 1->(0.01000)\n");
		initial_data(fp);
		backpropagation(fp);
		print_w(fp);
		fprintf(fp, "成功=%d,失敗=%d\n",ok,nook);
		b=(double)sumloop/(double)ok;
		fprintf(fp, "平均反復回数=%f\n",b);
	}
	atime = atime / (double)ok;
	fprintf(fp, "平均時間=%lf", atime);
	fclose(fp);
	return 0;
}
			


