/*
 * cblas_ztbmv.c
 * The program is a C interface to ztbmv.
 * 
 * Keita Teranishi  5/20/98
 *
 */
#include "cblas.h"
#include "cblas_f77.h"
void cblas_ztbmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const CBLAS_INT_TYPE N, const CBLAS_INT_TYPE K, const void  *A, const CBLAS_INT_TYPE lda,
                 void  *X, const CBLAS_INT_TYPE incX)
{
   char TA;
   char UL;
   char DI;
#ifdef F77_CHAR
   F77_CHAR F77_TA, F77_UL, F77_DI;
#else
   #define F77_TA &TA
   #define F77_UL &UL
   #define F77_DI &DI   
#endif
#ifdef F77_INT
   F77_INT F77_N=N, F77_lda=lda, F77_K=K, F77_incX=incX;
#else
   #define F77_N N
   #define F77_K K
   #define F77_lda lda
   #define F77_incX incX
#endif
   CBLAS_INT_TYPE n, i=0, tincX; 
   double *st=0, *x=(double *)X;

   if (order == CblasColMajor)
   {
      if (Uplo == CblasUpper) UL = 'U';
      else if (Uplo == CblasLower) UL = 'L';
      else 
      {
         cblas_xerbla(2, "cblas_ztbmv","Illegal Uplo setting, %d\n", Uplo);
         return;
      }
      if (TransA == CblasNoTrans) TA = 'N';
      else if (TransA == CblasTrans) TA = 'T';
      else if (TransA == CblasConjTrans) TA = 'C';
      else 
      {
         cblas_xerbla(3, "cblas_ztbmv","Illegal TransA setting, %d\n", TransA);
         return;
      }
      if (Diag == CblasUnit) DI = 'U';
      else if (Diag == CblasNonUnit) DI = 'N';
      else 
      {
         cblas_xerbla(4, "cblas_ztbmv","Illegal Diag setting, %d\n", Diag);
         return;
      }
      #ifdef F77_CHAR
         F77_UL = C2F_CHAR(&UL);
         F77_TA = C2F_CHAR(&TA);
         F77_DI = C2F_CHAR(&DI);
      #endif
      F77_ztbmv( F77_UL, F77_TA, F77_DI, &F77_N, &F77_K, A, &F77_lda, X,
                      &F77_incX);
   }
   else if (order == CblasRowMajor)
   {
      if (Uplo == CblasUpper) UL = 'L';
      else if (Uplo == CblasLower) UL = 'U';
      else 
      {
         cblas_xerbla(2, "cblas_ztbmv","Illegal Uplo setting, %d\n", Uplo);
         return;
      }

      if (TransA == CblasNoTrans) TA = 'T';
      else if (TransA == CblasTrans) TA = 'N';
      else if (TransA == CblasConjTrans)
      {
         TA = 'N';
         if ( N > 0)
         {
            if(incX > 0)
               tincX = incX;
            else
               tincX = -incX;
            i = tincX << 1;
            n = i * N;
            x++;
            st = x + n;
            do
            {
               *x = -(*x);
               x+= i;
            }
            while (x != st);
            x -= n;
         }
      }
      else 
      {
         cblas_xerbla(3, "cblas_ztbmv","Illegal TransA setting, %d\n", TransA);
         return;
      }

      if (Diag == CblasUnit) DI = 'U';
      else if (Diag == CblasNonUnit) DI = 'N';
      else 
      {
         cblas_xerbla(4, "cblas_ztbmv","Illegal Uplo setting, %d\n", Uplo);
         return;
      }
      #ifdef F77_CHAR
         F77_UL = C2F_CHAR(&UL);
         F77_TA = C2F_CHAR(&TA);
         F77_DI = C2F_CHAR(&DI);
      #endif

      F77_ztbmv( F77_UL, F77_TA, F77_DI, &F77_N, &F77_K, A, &F77_lda, X,
                      &F77_incX);

      if (TransA == CblasConjTrans)
      {
         if (N > 0)
         {
            do
            {
               *x = -(*x);
               x += i;
            }
            while (x != st);
         }
      }
   }
   else cblas_xerbla(1, "cblas_ztbmv", "Illegal Order setting, %d\n", order);
   return;
}
