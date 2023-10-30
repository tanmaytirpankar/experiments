#include <stdio.h>
#include <iostream>
#include <string>
#include <cmath>
#include <vector>
#include <algorithm>
#include <mpfr.h>

using namespace std;

// Termination criterion
const double NEARZERO = 1.0e-10;       // interpretation of "zero"

// Alternative termination criterion - Number of iterations
int iterations = 2;

// Precisions
// Higher precision
#define hp 64
// Lower precision
#define lp 16

// Dimension of the problem
const u_int N = 2;

using vec    = mpfr_t*;         // vector
using matrix = vec*;            // matrix (=collection of (row) vectors)

FILE *data_file_hp;
FILE *data_file_lp;

// Prototypes
void print( const string& title, const vec &V );
void print( const string& title, const matrix &A );
void writeToFile(const vec &V, FILE *data_file);
vec matrixTimesVector( const matrix &A, const vec &V );
vec vectorCombination( mpfr_t& a, const vec &U, mpfr_t& b, const vec &V );
mpfr_t* innerProduct( const vec &U, const vec &V );
mpfr_t* vectorNorm( const vec &V );
double cosine_similarity(const vec &U, const vec &V);
//float rand_f();
vec conjugateGradientSolver( const matrix &A_hp, const vec &B_hp,
                             const matrix &A_lp, const vec &B_lp );

//======================================================================


int main() {
  // seeds the generator
//  srand(time(0));

  auto A_hp = new vec[2];
  A_hp[0] = new mpfr_t[2];
  A_hp[1] = new mpfr_t[2];
  mpfr_init2(A_hp[0][0], hp);
  mpfr_init2(A_hp[0][1], hp);
  mpfr_init2(A_hp[1][0], hp);
  mpfr_init2(A_hp[1][1], hp);
  mpfr_set_d(A_hp[0][0], 3.96751, MPFR_RNDN);
  mpfr_set_d(A_hp[0][1], 1.59033, MPFR_RNDN);
  mpfr_set_d(A_hp[1][0], 0.72374, MPFR_RNDN);
  mpfr_set_d(A_hp[1][1], 1.59008, MPFR_RNDN);

  vec B_hp = new mpfr_t[2];
  mpfr_init2(B_hp[0], hp);
  mpfr_init2(B_hp[1], hp);
  mpfr_set_d(B_hp[0], 1.19822, MPFR_RNDN);
  mpfr_set_d(B_hp[1], 0.359596, MPFR_RNDN);

  auto A_lp = new vec[2];
  A_lp[0] = new mpfr_t[2];
  A_lp[1] = new mpfr_t[2];
  mpfr_init2(A_lp[0][0], lp);
  mpfr_init2(A_lp[0][1], lp);
  mpfr_init2(A_lp[1][0], lp);
  mpfr_init2(A_lp[1][1], lp);
  mpfr_set_d(A_lp[0][0], 3.96751, MPFR_RNDN);
  mpfr_set_d(A_lp[0][1], 1.59033, MPFR_RNDN);
  mpfr_set_d(A_lp[1][0], 0.72374, MPFR_RNDN);
  mpfr_set_d(A_lp[1][1], 1.59008, MPFR_RNDN);

  vec B_lp = new mpfr_t[2];
  mpfr_init2(B_lp[0], lp);
  mpfr_init2(B_lp[1], lp);
  mpfr_set_d(B_lp[0], 1.19822, MPFR_RNDN);
  mpfr_set_d(B_lp[1], 0.359596, MPFR_RNDN);


  data_file_hp = fopen("data_hp.csv", "w");
  data_file_lp = fopen("data_lp.csv", "w");

  cout << "Solves AX = B_hp\n";
  print( "\nA_hp:", A_hp );
  print( "\nB_hp:", B_hp );

  vec X = conjugateGradientSolver( A_hp, B_hp, A_lp, B_lp );

  cout << "\nSolution:";
  print( "\nX:", X );
  print( "\nCheck AX:", matrixTimesVector( A_hp, X ) );

  fclose(data_file_hp);
  fclose(data_file_lp);

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
void writeToFile(const vec &V, FILE *data_file) {
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
vec conjugateGradientSolver( const matrix &A_hp, const vec &B_hp,
                             const matrix &A_lp, const vec &B_lp ) {
  auto *TOLERANCE = new mpfr_t[1];
  mpfr_init2(TOLERANCE[0], hp);
  mpfr_set_d(TOLERANCE[0], 1.0e-10, MPFR_RNDN);

  // Initializing vector X which will be set to the solution by the end of this algorithm.
  vec X_hp = new mpfr_t[N];
  for (int i = 0; i < N; i++) {
    mpfr_init2(X_hp[i], hp);
    mpfr_set_d(X_hp[i], 0.0, MPFR_RNDN);
  }
  vec X_lp = new mpfr_t[N];
  for (int i = 0; i < N; i++) {
    mpfr_init2(X_lp[i], lp);
    mpfr_set_d(X_lp[i], 0.0, MPFR_RNDN);
  }

  vec R_hp = new mpfr_t[N];
  for (int i = 0; i < N; i++) {
    mpfr_init2(R_hp[i], hp);
    mpfr_set(R_hp[i], B_hp[i], MPFR_RNDN);
  }
  vec R_lp = new mpfr_t[N];
  for (int i = 0; i < N; i++) {
    mpfr_init2(R_lp[i], lp);
    mpfr_set(R_lp[i], B_lp[i], MPFR_RNDN);
  }

  vec P_hp = new mpfr_t[N];
  for (int i = 0; i < N; i++) {
    mpfr_init2(P_hp[i], hp);
    mpfr_set(P_hp[i], R_hp[i], MPFR_RNDN);
  }
  vec P_lp = new mpfr_t[N];
  for (int i = 0; i < N; i++) {
    mpfr_init2(P_lp[i], lp);
    mpfr_set(P_lp[i], R_lp[i], MPFR_RNDN);
  }

  int k = 0;

  writeToFile(X_hp, data_file_hp);
  writeToFile(X_lp, data_file_lp);

  while (k < iterations) {
    cout << "\nIteration " << k;
    print("\nX_hp:", X_hp);
    print("\nX_lp:", X_lp);

    vec Rold_hp = new mpfr_t[N];
    for (int i = 0; i < N; i++) {
      mpfr_init2(Rold_hp[i], hp);
      mpfr_set(Rold_hp[i], R_hp[i], MPFR_RNDN);
    }
    vec Rold_lp = new mpfr_t[N];
    for (int i = 0; i < N; i++) {
      mpfr_init2(Rold_lp[i], lp);
      mpfr_set(Rold_lp[i], R_lp[i], MPFR_RNDN);
    }

    vec AP_hp = matrixTimesVector(A_hp, P_hp);
    vec AP_lp = matrixTimesVector(A_lp, P_lp);
//    print("\nAP:", AP);

    auto *NEARZERO_MPFR_hp = new mpfr_t[1];
    mpfr_init2(*NEARZERO_MPFR_hp, hp);
    mpfr_set_d(*NEARZERO_MPFR_hp, NEARZERO, MPFR_RNDN);
    auto *NEARZERO_MPFR_lp = new mpfr_t[1];
    mpfr_init2(*NEARZERO_MPFR_lp, lp);
    mpfr_set_d(*NEARZERO_MPFR_lp, NEARZERO, MPFR_RNDN);

    auto *max_hp = new mpfr_t[1];
    mpfr_init2(*max_hp, hp);
    mpfr_max(*max_hp, *innerProduct(P_hp, AP_hp), *NEARZERO_MPFR_hp, MPFR_RNDN);
    auto *max_lp = new mpfr_t[1];
    mpfr_init2(*max_lp, lp);
    mpfr_max(*max_lp, *innerProduct(P_lp, AP_lp), *NEARZERO_MPFR_lp, MPFR_RNDN);

    auto *alpha_hp = new mpfr_t[1];
    mpfr_init2(alpha_hp[0], hp);
    mpfr_div(alpha_hp[0], *innerProduct(R_hp, R_hp), *max_hp, MPFR_RNDN);
    auto *alpha_lp = new mpfr_t[1];
    mpfr_init2(alpha_lp[0], lp);
    mpfr_div(alpha_lp[0], *innerProduct(R_lp, R_lp), *max_lp, MPFR_RNDN);
//    print("\nalpha:", *alpha);

    auto *ZERO_POINT_ZERO_hp = new mpfr_t[1];
    mpfr_init2(ZERO_POINT_ZERO_hp[0], hp);
    mpfr_set_d(ZERO_POINT_ZERO_hp[0], 0.0, MPFR_RNDN);
    cout << "Cosine Similarity: " <<
         cosine_similarity(vectorCombination(*ZERO_POINT_ZERO_hp, X_hp, *alpha_hp, P_hp),
                           vectorCombination(*ZERO_POINT_ZERO_hp, X_lp, *alpha_lp, P_lp)) << endl;

    auto *ONE_POINT_ZERO_hp = new mpfr_t[1];
    mpfr_init2(ONE_POINT_ZERO_hp[0], hp);
    mpfr_set_d(ONE_POINT_ZERO_hp[0], 1.0, MPFR_RNDN);
    auto *ONE_POINT_ZERO_lp = new mpfr_t[1];
    mpfr_init2(ONE_POINT_ZERO_lp[0], lp);
    mpfr_set_d(ONE_POINT_ZERO_lp[0], 1.0, MPFR_RNDN);

    auto *alpha_neg_hp = new mpfr_t[1];
    mpfr_init2(alpha_neg_hp[0], hp);
    mpfr_neg(alpha_neg_hp[0], alpha_hp[0], MPFR_RNDN);
    auto *alpha_neg_lp = new mpfr_t[1];
    mpfr_init2(alpha_neg_lp[0], lp);
    mpfr_neg(alpha_neg_lp[0], alpha_lp[0], MPFR_RNDN);

    X_hp = vectorCombination(*ONE_POINT_ZERO_hp, X_hp, *alpha_hp, P_hp);            // Next estimate of solution
    X_lp = vectorCombination(*ONE_POINT_ZERO_lp, X_lp, *alpha_lp, P_lp);
    R_hp = vectorCombination(*ONE_POINT_ZERO_hp, R_hp, *alpha_neg_hp, AP_hp);          // Residual
    R_lp = vectorCombination(*ONE_POINT_ZERO_lp, R_lp, *alpha_neg_lp, AP_lp);
//    print("\nX_hp:", X_hp);
//    print("\nR:", R);

    writeToFile(X_hp, data_file_hp);
    writeToFile(X_lp, data_file_lp);

    mpfr_t *R_norm_hp = vectorNorm(R_hp);
    mpfr_sqrt(R_norm_hp[0], R_norm_hp[0], MPFR_RNDN);
    if (mpfr_cmp(R_norm_hp[0], TOLERANCE[0]) < 0) {
      cout << "\nConverged in " << k << " iterations.\n";
      break;             // Convergence test
    }

    auto *beta_hp = new mpfr_t[1];
    mpfr_init2(beta_hp[0], hp);
    mpfr_max(*max_hp, *innerProduct(Rold_hp, Rold_hp), *NEARZERO_MPFR_hp, MPFR_RNDN);
    auto *beta_lp = new mpfr_t[1];
    mpfr_init2(beta_lp[0], lp);
    mpfr_max(*max_lp, *innerProduct(Rold_lp, Rold_lp), *NEARZERO_MPFR_lp, MPFR_RNDN);

    mpfr_div(beta_hp[0], *innerProduct(R_hp, R_hp), *max_hp, MPFR_RNDN);
    mpfr_div(beta_lp[0], *innerProduct(R_lp, R_lp), *max_lp, MPFR_RNDN);
//    print("\nbeta:", *beta_hp);

    P_hp = vectorCombination(*ONE_POINT_ZERO_hp, R_hp, *beta_hp, P_hp);             // Next gradient
    P_lp = vectorCombination(*ONE_POINT_ZERO_lp, R_lp, *beta_lp, P_lp);
    k++;
  }

  if (k == iterations) {
    cout << "\nDid not converge in " << k << " iterations.\n";
  }

  cout << endl;

  return X_hp;
}
