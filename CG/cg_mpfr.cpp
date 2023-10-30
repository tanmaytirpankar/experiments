#include <stdio.h>
#include <iostream>
#include <string>
#include <cmath>
#include <vector>
#include <algorithm>
#include <mpfr.h>
using namespace std;

// Termination criterion
const float NEARZERO = 1.0e-10;       // interpretation of "zero"

// Alternative termination criterion - Number of iterations
int iterations = 2;

// Precision
#define prec 16

// Dimension of the problem
const u_int N = 2;

using vec    = mpfr_t*;         // vector
using matrix = vec*;            // matrix (=collection of (row) vectors)

FILE *data_file;

// Prototypes
void print( const string& title, const vec &V );
void print( const string& title, const matrix &A );
void writeToFile(const vec &V);
vec matrixTimesVector( const matrix &A, const vec &V );
vec vectorCombination( mpfr_t& a, const vec &U, mpfr_t& b, const vec &V );
mpfr_t* innerProduct( const vec &U, const vec &V );
mpfr_t* vectorNorm( const vec &V );
double cosine_similarity(const vec &U, const vec &V);
//float rand_f();
vec conjugateGradientSolver( const matrix &A, const vec &B );

//======================================================================


int main() {
  // seeds the generator
//  srand(time(0));

  auto A = new vec[2];
  A[0] = new mpfr_t[2];
  A[1] = new mpfr_t[2];
  mpfr_init2(A[0][0], prec);
  mpfr_init2(A[0][1], prec);
  mpfr_init2(A[1][0], prec);
  mpfr_init2(A[1][1], prec);
  mpfr_set_d(A[0][0], 3.96751, MPFR_RNDN);
  mpfr_set_d(A[0][1], 1.59033, MPFR_RNDN);
  mpfr_set_d(A[1][0], 0.72374, MPFR_RNDN);
  mpfr_set_d(A[1][1], 1.59008, MPFR_RNDN);

  vec B = new mpfr_t[2];
  mpfr_init2(B[0], prec);
  mpfr_init2(B[1], prec);
  mpfr_set_d(B[0], 1.19822, MPFR_RNDN);
  mpfr_set_d(B[1], 0.359596, MPFR_RNDN);


  data_file = fopen("data.csv", "w");

  cout << "Solves AX = B\n";
  print( "\nA:", A );
  print( "\nB:", B );

  vec X = conjugateGradientSolver( A, B );

  cout << "\nSolution:";
  print( "\nX:", X );
  print( "\nCheck AX:", matrixTimesVector( A, X ) );

  fclose(data_file);

  return 0;
}


//======================================================================
// Prints the mpfr_t V
void print( const string& title, const mpfr_t& V )
{
  cout << title << '\n';

  mpfr_out_str(stdout, 10, 0, V, MPFR_RNDN);
  cout << '\n';
}

// Prints the vector V
void print( const string& title, const vec &V )
{
  cout << title << '\n';

  for ( int i = 0; i < N; i++ ) {
    mpfr_out_str(stdout, 10, 0, V[i], MPFR_RNDN);
    cout << "\t";
  }
  cout << '\n';
}


//======================================================================

// Prints the matrix A
void print( const string& title, const matrix &A )
{
  cout << title << '\n';
  // A is an m x n matrix
  for ( int i = 0; i < N; i++ ) {
    for ( int j = 0; j < N; j++ ) {
        mpfr_out_str(stdout, 10, 0, A[i][j], MPFR_RNDN);
        cout << "\t";
    }
    cout << '\n';
  }
}


//======================================================================

// Writes the vector V to a file
void writeToFile(const vec &V) {
  mpfr_out_str(data_file, 10, 0, V[0], MPFR_RNDN);
  for (int i = 1; i < N; i++) {
    fprintf(data_file, ",");
    mpfr_out_str(data_file, 10, 0, V[i], MPFR_RNDN);
  }
  fprintf(data_file, "\n");
}

//======================================================================

// Inner product of the matrix A with vector V returned as C (a vector)
vec matrixTimesVector( const matrix &A, const vec &V )     // Matrix times vector
{
    vec C = new mpfr_t[N];
    for ( int i = 0; i < N; i++ ) {
        mpfr_init2(C[i], V[0]->_mpfr_prec);
        mpfr_t temp;
        mpfr_init2(temp, V[0]->_mpfr_prec);
        mpfr_set_d(temp, 0.0, MPFR_RNDN);
        for ( int j = 0; j < N; j++ ) {
            mpfr_t temp2;
            mpfr_init2(temp2, V[0]->_mpfr_prec);
            mpfr_mul(temp2, A[i][j], V[j], MPFR_RNDN);
            mpfr_add(temp, temp, temp2, MPFR_RNDN);
        }
        mpfr_set(C[i], temp, MPFR_RNDN);
    }
//    print("C:", C);
    return C;
}


//======================================================================

// Returns the Linear combination of aU+bV as a vector W.
vec vectorCombination( mpfr_t& a, const vec &U, mpfr_t& b, const vec &V )        // Linear combination of vectors
{
    vec W = new mpfr_t[N];
    for ( int j = 0; j < N; j++ ) {
        mpfr_init2(W[j], V[0]->_mpfr_prec);
        mpfr_set_d(W[j], 0.0, MPFR_RNDN);
        mpfr_t temp;
        mpfr_init2(temp, V[0]->_mpfr_prec);
        mpfr_mul(temp, a, U[j], MPFR_RNDN);
        mpfr_mul(temp, b, V[j], MPFR_RNDN);
        mpfr_add(W[j], temp, W[j], MPFR_RNDN);
    }

    return W;
}


//======================================================================

// Returns the inner product of vector U with V.
mpfr_t* innerProduct( const vec &U, const vec &V )          // Inner product of U and V
{
    auto* C = new mpfr_t[1];
    mpfr_init2(C[0], V[0]->_mpfr_prec);
    mpfr_set_d(C[0], 0.0, MPFR_RNDN);
    for ( int j = 0; j < N; j++ ) {
        mpfr_t temp;
        mpfr_init2(temp, V[0]->_mpfr_prec);
        mpfr_mul(temp, U[j], V[j], MPFR_RNDN);
        mpfr_add(C[0], C[0], temp, MPFR_RNDN);
    }

    return C;
}


//======================================================================

// Computes and returns the Euclidean/2-norm of the vector V.
mpfr_t* vectorNorm( const vec &V )                          // Vector norm
{
    return innerProduct( V, V );
}


//======================================================================

// Computes and returns the cosine similarity of the vector U and V.
double cosine_similarity(const vec &U, const vec &V) {
  mpfr_t* inner = innerProduct(U, V);
  mpfr_t* sqr_normU = vectorNorm(U);
  mpfr_t* normU = new mpfr_t[1];
  mpfr_init2(normU[0], U[0]->_mpfr_prec);
  mpfr_sqrt(normU[0], sqr_normU[0], MPFR_RNDN);
  mpfr_t* sqr_normV = vectorNorm(V);
  mpfr_t* normV = new mpfr_t[1];
  mpfr_init2(normV[0], V[0]->_mpfr_prec);
  mpfr_sqrt(normV[0], sqr_normV[0], MPFR_RNDN);

  auto* norm_prod = new mpfr_t[1];
  mpfr_init2(norm_prod[0], U[0]->_mpfr_prec);
  mpfr_mul(norm_prod[0], normU[0], normV[0], MPFR_RNDN);

  auto* normUV = new mpfr_t[1];
  mpfr_init2(normUV[0], U[0]->_mpfr_prec);
  mpfr_div(normUV[0], inner[0], norm_prod[0], MPFR_RNDN);

  return mpfr_get_d(normUV[0], MPFR_RNDN);
}

//======================================================================

// Returns a random float
//float rand_f()
//{
//  return (float)(rand()) / (float)(rand());
//}

//======================================================================

// The conjugate gradient solving algorithm.
vec conjugateGradientSolver( const matrix &A, const vec &B ) {
  auto *TOLERANCE = new mpfr_t[1];
  mpfr_init2(TOLERANCE[0], prec);
  mpfr_set_d(TOLERANCE[0], 1.0e-10, MPFR_RNDN);

  // Initializing vector X which will be set to the solution by the end of this algorithm.
  vec X = new mpfr_t[N];
  for (int i = 0; i < N; i++) {
    mpfr_init2(X[i], prec);
    mpfr_set_d(X[i], 0.0, MPFR_RNDN);
  }

  vec R = new mpfr_t[N];
  for (int i = 0; i < N; i++) {
    mpfr_init2(R[i], prec);
    mpfr_set(R[i], B[i], MPFR_RNDN);
  }
  vec P = new mpfr_t[N];
  for (int i = 0; i < N; i++) {
    mpfr_init2(P[i], prec);
    mpfr_set(P[i], R[i], MPFR_RNDN);
  }

  int k = 0;

  writeToFile(X);

  while (k < iterations) {
    cout << "\nIteration " << k;
    print("\nX:", X);

    vec Rold = new mpfr_t[N];
    for (int i = 0; i < N; i++) {
      mpfr_init2(Rold[i], prec);
      mpfr_set(Rold[i], R[i], MPFR_RNDN);
    }
    vec AP = matrixTimesVector(A, P);
//    print("\nAP:", AP);

    auto *NEARZERO_MPFR = new mpfr_t[1];
    mpfr_init2(*NEARZERO_MPFR, prec);
    mpfr_set_d(*NEARZERO_MPFR, NEARZERO, MPFR_RNDN);

    auto *max = new mpfr_t[1];
    mpfr_init2(*max, prec);
    mpfr_max(*max, *innerProduct(P, AP), *NEARZERO_MPFR, MPFR_RNDN);

    auto *alpha = new mpfr_t[1];
    mpfr_init2(alpha[0], prec);
    mpfr_div(alpha[0], *innerProduct(R, R), *max, MPFR_RNDN);
//    print("\nalpha:", *alpha);

    auto *ONE_POINT_ZERO = new mpfr_t[1];
    mpfr_init2(ONE_POINT_ZERO[0], prec);
    mpfr_set_d(ONE_POINT_ZERO[0], 1.0, MPFR_RNDN);
    auto *alpha_neg = new mpfr_t[1];
    mpfr_init2(alpha_neg[0], prec);
    mpfr_neg(alpha_neg[0], alpha[0], MPFR_RNDN);
    X = vectorCombination(*ONE_POINT_ZERO, X, *alpha, P);            // Next estimate of solution
    R = vectorCombination(*ONE_POINT_ZERO, R, *alpha_neg, AP);          // Residual
//    print("\nX:", X);
//    print("\nR:", R);

    writeToFile(X);

    mpfr_t *R_norm = vectorNorm(R);
    if (mpfr_cmp(R_norm[0], TOLERANCE[0]) < 0) {
      cout << "\nConverged in " << k << " iterations.\n";
      break;             // Convergence test
    }

    auto *beta = new mpfr_t[1];
    mpfr_init2(beta[0], prec);
    mpfr_max(*max, *innerProduct(Rold, Rold), *NEARZERO_MPFR, MPFR_RNDN);
    mpfr_div(beta[0], *innerProduct(R, R), *max, MPFR_RNDN);
//    print("\nbeta:", *beta);

    P = vectorCombination(*ONE_POINT_ZERO, R, *beta, P);             // Next gradient
    k++;
  }

  if (k == iterations) {
    cout << "\nDid not converge in " << k << " iterations.\n";
  }

  cout << endl;

  return X;
}

