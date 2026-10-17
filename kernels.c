// CPU kernel study: matmul (naive / transposed-B / tiled / tiled+AVX2 FMA) and sum reduction
// (scalar / 8-accumulator / AVX2). Written with AI assistance under Jacob Garcia's direction.
#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static void mm_naive(const float*A,const float*B,float*C,int n){
 for(int i=0;i<n;i++)for(int j=0;j<n;j++){float s=0;for(int k=0;k<n;k++)s+=A[i*n+k]*B[k*n+j];C[i*n+j]=s;}}
#define BS 64
static void mm_tiled(const float*A,const float*B,float*C,int n){
 memset(C,0,sizeof(float)*n*n);
 for(int ii=0;ii<n;ii+=BS)for(int kk=0;kk<n;kk+=BS)for(int jj=0;jj<n;jj+=BS)
  for(int i=ii;i<ii+BS;i++)for(int k=kk;k<kk+BS;k++){float a=A[i*n+k];
   for(int j=jj;j<jj+BS;j++)C[i*n+j]+=a*B[k*n+j];}}
__attribute__((target("avx2,fma")))
static void mm_avx2(const float*A,const float*B,float*C,int n){
 memset(C,0,sizeof(float)*n*n);
 for(int ii=0;ii<n;ii+=BS)for(int kk=0;kk<n;kk+=BS)for(int jj=0;jj<n;jj+=BS)
  for(int i=ii;i<ii+BS;i++)for(int k=kk;k<kk+BS;k++){__m256 a=_mm256_set1_ps(A[i*n+k]);
   for(int j=jj;j<jj+BS;j+=8){__m256 c=_mm256_loadu_ps(&C[i*n+j]);
    c=_mm256_fmadd_ps(a,_mm256_loadu_ps(&B[k*n+j]),c);_mm256_storeu_ps(&C[i*n+j],c);}}}
static float sum_scalar(const float*x,long n){float s=0;
 #pragma GCC novector
 for(long i=0;i<n;i++)s+=x[i];return s;}
static float sum_acc8(const float*x,long n){float s[8]={0};for(long i=0;i<n;i+=8)for(int j=0;j<8;j++)s[j]+=x[i+j];
 float t=0;for(int j=0;j<8;j++)t+=s[j];return t;}
__attribute__((target("avx2")))
static float sum_avx2(const float*x,long n){__m256 a0=_mm256_setzero_ps(),a1=a0,a2=a0,a3=a0;
 for(long i=0;i<n;i+=32){a0=_mm256_add_ps(a0,_mm256_loadu_ps(x+i));a1=_mm256_add_ps(a1,_mm256_loadu_ps(x+i+8));
  a2=_mm256_add_ps(a2,_mm256_loadu_ps(x+i+16));a3=_mm256_add_ps(a3,_mm256_loadu_ps(x+i+24));}
 a0=_mm256_add_ps(_mm256_add_ps(a0,a1),_mm256_add_ps(a2,a3));float t[8];_mm256_storeu_ps(t,a0);
 float s=0;for(int j=0;j<8;j++)s+=t[j];return s;}
typedef void(*mmf)(const float*,const float*,float*,int);
static double bench_mm(mmf f,const float*A,const float*B,float*C,int n,int reps){double best=1e9;
 for(int r=0;r<reps;r++){double t=now();f(A,B,C,n);t=now()-t;if(t<best)best=t;}return best;}
int main(){
 int n=1024; size_t N=(size_t)n*n;
 float*A=aligned_alloc(64,N*4),*B=aligned_alloc(64,N*4),*C=aligned_alloc(64,N*4),*R=aligned_alloc(64,N*4);
 srand(1);for(size_t i=0;i<N;i++){A[i]=rand()/(float)RAND_MAX-.5f;B[i]=rand()/(float)RAND_MAX-.5f;}
 struct{const char*name;mmf f;int reps;}M[]={{"naive ijk",mm_naive,2},{"tiled 64 (scalar C)",mm_tiled,5},{"tiled 64 + AVX2 FMA",mm_avx2,8}};
 double flop=2.0*n*n*n; double base=0;
 printf("MATMUL n=%d fp32 (best of reps)\n",n);
 for(int m=0;m<3;m++){double t=bench_mm(M[m].f,A,B,C,n,M[m].reps);
  if(m==0){memcpy(R,C,N*4);base=t;}
  double err=0;for(size_t i=0;i<N;i++){double e=fabs(C[i]-R[i]);if(e>err)err=e;}
  printf("  %-22s %8.3f s  %7.2f GFLOP/s  speedup %6.2fx  max|diff vs naive| %.2e\n",M[m].name,t,flop/t/1e9,base/t,err);}
 long L=1L<<26; float*x=aligned_alloc(64,L*4);for(long i=0;i<L;i++)x[i]=(i%7)*0.01f;
 double ref=0;for(long i=0;i<L;i++)ref+=x[i];
 printf("REDUCTION sum of %ld fp32 (%.0f MB)\n",L,L*4/1e6);
 struct{const char*n;float(*f)(const float*,long);}S[]={{"scalar",sum_scalar},{"8 accumulators",sum_acc8},{"AVX2 4x8",sum_avx2}};
 double bs=0;for(int s=0;s<3;s++){double best=1e9;float v=0;for(int r=0;r<8;r++){double t=now();v=S[s].f(x,L);t=now()-t;if(t<best)best=t;}
  if(s==0)bs=best;printf("  %-16s %7.4f s  %6.2f GB/s  speedup %5.2fx  rel.err vs double %.2e\n",S[s].n,best,L*4/best/1e9,bs/best,fabs(v-ref)/ref);}
 return 0;}
