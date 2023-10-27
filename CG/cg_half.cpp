#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <mpfr.h>
using namespace std;

const float NEARZERO = 1.0e-10;       // interpretation of "zero"
const u_int N = 2;

using vec    = mpfr_t*;         // vector
using matrix = vec*;            // matrix (=collection of (row) vectors)

//ofstream data_file;

// Prototypes
void print( string title, const vec &V );
void print( string title, const matrix &A );
//void writeToFile(const vec &V);
vec matrixTimesVector( const matrix &A, const vec &V );
vec vectorCombination( mpfr_t a, const vec &U, mpfr_t b, const vec &V );
mpfr_t* innerProduct( const vec &U, const vec &V );
mpfr_t* vectorNorm( const vec &V );
double cosine_similarity(const vec &U, const vec &V);
//float rand_f();
vec conjugateGradientSolver( const matrix &A, const vec &B );

//======================================================================


int main() {
  // seeds the generator
//  srand(time(0));

    vec B = new mpfr_t[2];
    mpfr_init2(B[0], 16);
    mpfr_init2(B[1], 16);
    mpfr_set_d(B[0], 1.19822, MPFR_RNDN);
    mpfr_set_d(B[1], 0.359596, MPFR_RNDN);

    matrix A = new vec[2];
    A[0] = new mpfr_t[2];
    A[1] = new mpfr_t[2];
    mpfr_init2(A[0][0], 16);
    mpfr_init2(A[0][1], 16);
    mpfr_init2(A[1][0], 16);
    mpfr_init2(A[1][1], 16);
    mpfr_set_d(A[0][0], 3.96751, MPFR_RNDN);
    mpfr_set_d(A[0][1], 1.59033, MPFR_RNDN);
    mpfr_set_d(A[1][0], 0.72374, MPFR_RNDN);
    mpfr_set_d(A[1][1], 1.59008, MPFR_RNDN);


//  matrix A = { { rand_f(), rand_f() }, { rand_f(), rand_f() } };
//  vec B = { rand_f(), rand_f() };

//  matrix A = { { 3.96751, 1.59033 }, { 0.72374, 1.59008 } };
//  vec B = { 1.19822, 0.359596 };
//  mpfr_inits(A, 3.96751, 1.59033)
//
//  data_file.open ("data_float.csv");
//
  cout << "Solves AX = B\n";
  print( "\nA:", A );
  print( "\nB:", B );
//
//  vec X = conjugateGradientSolver( A, B );
//
//  cout << "\nSolution:";
//  print( "\nX:", X );
//  print( "\nCheck AX:", matrixTimesVector( A, X ) );
//
//  data_file.close();

  return 0;
}


//======================================================================

// Prints the vector V
void print( string title, const vec &V )
{
  cout << title << '\n';

  for ( int i = 0; i < N; i++ ) {
      mpfr_out_str(stdout, 10, 0, V[i], MPFR_RNDN);
  }
  cout << '\n';
}


//======================================================================

// Prints the matrix A
void print( string title, const matrix &A )
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
//void writeToFile(const vec &V) {
//  data_file << V[0];
//  for (int i = 1; i < V.size(); i++) {
//    data_file << "," << V[i];
//  }
//  data_file << endl;
//}

//======================================================================

// Inner product of the matrix A with vector V returned as C (a vector)
vec matrixTimesVector( const matrix &A, const vec &V )     // Matrix times vector
{
    vec C = new mpfr_t[N];
    for ( int i = 0; i < N; i++ ) {
        mpfr_t temp;
        mpfr_init2(temp, 16);
        mpfr_set_d(temp, 0.0, MPFR_RNDN);
        for ( int j = 0; j < N; j++ ) {
            mpfr_t temp2;
            mpfr_init2(temp2, 16);
            mpfr_mul(temp2, A[i][j], V[j], MPFR_RNDN);
            mpfr_add(temp, temp, temp2, MPFR_RNDN);
        }
        mpfr_set(C[i], temp, MPFR_RNDN);
    }
    return C;
}


//======================================================================

// Returns the Linear combination of aU+bV as a vector W.
vec vectorCombination( mpfr_t& a, const vec &U, mpfr_t& b, const vec &V )        // Linear combination of vectors
{
    vec W = new mpfr_t[N];
    for ( int j = 0; j < N; j++ ) {
        mpfr_t temp;
        mpfr_init2(temp, 16);
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
    mpfr_t* C = new mpfr_t[1];
    mpfr_init2(C[0], 16);
    mpfr_set_d(C[0], 0.0, MPFR_RNDN);
    for ( int j = 0; j < N; j++ ) {
        mpfr_t temp;
        mpfr_init2(temp, 16);
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
    mpfr_t* normU = vectorNorm(U);
    mpfr_t* normV = vectorNorm(V);
    mpfr_t* normUV = new mpfr_t[1];
    mpfr_init2(normUV[0], 16);
    mpfr_sqrt(normUV[0], normU[0], MPFR_RNDN);
    mpfr_sqrt(normUV[0], normV[0], MPFR_RNDN);
    mpfr_div(normUV[0], inner[0], normUV[0], MPFR_RNDN);
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
vec conjugateGradientSolver( const matrix &A, const vec &B )
{
//  // Setting a tolerance level which will be used as a termination condition for this algorithm
//  float TOLERANCE = 1.0e-10;
//
//  // Number of vectors/rows in the matrix A.
//  int n = A.size();
//
//  // Initializing vector X which will be set to the solution by the end of this algorithm.
//  vec X( n, 0.0 );
//
//
//  vec R = B;
//  vec P = R;
//  int k = 0;
//
//  writeToFile(X);
//
//  while ( k < 10 ) {
//    cout << "\nIteration " << k;
//    print( "\nX:", X );
//
//    vec Rold = R;                                         // Store previous residual
//    vec AP = matrixTimesVector( A, P );
//
//    //
//    float alpha = innerProduct( R, R ) / max( innerProduct( P, AP ), NEARZERO );
//    X = vectorCombination( 1.0, X, alpha, P );            // Next estimate of solution
//    R = vectorCombination( 1.0, R, -alpha, AP );          // Residual
//    writeToFile(X);
//
//    if ( vectorNorm( R ) < TOLERANCE ) {
//      cout << "\nConverged in " << k << " iterations.\n";
//      break;             // Convergence test
//    }
//
//    float beta = innerProduct( R, R ) / max( innerProduct( Rold, Rold ), NEARZERO );
//    P = vectorCombination( 1.0, R, beta, P );             // Next gradient
//    k++;
//  }
//
//  if(k == n) {
//    cout << "\nDid not converge in " << k << " iterations.\n";
//  }
//
//  cout << endl;
//
//  return X;
}

