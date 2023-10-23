#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
using namespace std;

const float NEARZERO = 1.0e-10;       // interpretation of "zero"

using vec    = vector<float>;         // vector
using matrix = vector<vec>;            // matrix (=collection of (row) vectors)

ofstream data_file;

// Prototypes
void print( string title, const vec &V );
void print( string title, const matrix &A );
void writeToFile(const vec &V);
vec matrixTimesVector( const matrix &A, const vec &V );
vec vectorCombination( float a, const vec &U, float b, const vec &V );
float innerProduct( const vec &U, const vec &V );
float vectorNorm( const vec &V );
double cosine_similarity(const vec &U, const vec &V);
float rand_f();
vec conjugateGradientSolver( const matrix &A, const vec &B );

//======================================================================


int main() {
  // seeds the generator
  srand(time(0));

//  matrix A = { { rand_f(), rand_f() }, { rand_f(), rand_f() } };
//  vec B = { rand_f(), rand_f() };

  matrix A = { { 3.96751, 1.59033 }, { 0.72374, 1.59008 } };
  vec B = { 1.19822, 0.359596 };

  data_file.open ("data_float.csv");

  cout << "Solves AX = B\n";
  print( "\nA:", A );
  print( "\nB:", B );

  vec X = conjugateGradientSolver( A, B );

  cout << "\nSolution:";
  print( "\nX:", X );
  print( "\nCheck AX:", matrixTimesVector( A, X ) );

  data_file.close();
  return 0;
}


//======================================================================

// Prints the vector V
void print( string title, const vec &V )
{
  cout << title << '\n';

  int n = V.size();
  for ( int i = 0; i < n; i++ )
  {
    float x = V[i];   if ( abs( x ) < NEARZERO ) x = 0.0;
    cout << x << '\t';
  }
  cout << '\n';
}


//======================================================================

// Prints the matrix A
void print( string title, const matrix &A )
{
  cout << title << '\n';

  int m = A.size(), n = A[0].size();                      // A is an m x n matrix
  for ( int i = 0; i < m; i++ )
  {
    for ( int j = 0; j < n; j++ )
    {
      float x = A[i][j];   if ( abs( x ) < NEARZERO ) x = 0.0;
      cout << x << '\t';
    }
    cout << '\n';
  }
}


//======================================================================

// Writes the vector V to a file
void writeToFile(const vec &V) {
  data_file << V[0];
  for (int i = 1; i < V.size(); i++) {
    data_file << "," << V[i];
  }
  data_file << endl;
}

//======================================================================

// Inner product of the matrix A with vector V returned as C (a vector)
vec matrixTimesVector( const matrix &A, const vec &V )     // Matrix times vector
{
  int n = A.size();
  vec C( n );
  for ( int i = 0; i < n; i++ ) C[i] = innerProduct( A[i], V );
  return C;
}


//======================================================================

// Returns the Linear combination of aU+bV as a vector W.
vec vectorCombination( float a, const vec &U, float b, const vec &V )        // Linear combination of vectors
{
  int n = U.size();
  vec W( n );
  for ( int j = 0; j < n; j++ ) W[j] = a * U[j] + b * V[j];
  return W;
}


//======================================================================

// Returns the inner product of vector U with V.
float innerProduct( const vec &U, const vec &V )          // Inner product of U and V
{
  return inner_product( U.begin(), U.end(), V.begin(), 0.0 );
}


//======================================================================

// Computes and returns the Euclidean/2-norm of the vector V.
float vectorNorm( const vec &V )                          // Vector norm
{
  return sqrt( innerProduct( V, V ) );
}


//======================================================================

// Computes and returns the cosine similarity of the vector U and V.
double cosine_similarity(const vec &U, const vec &V) {
  return innerProduct(U, V) / (vectorNorm(U) * vectorNorm(V));
}

//======================================================================

// Returns a random float
float rand_f()
{
  return (float)(rand()) / (float)(rand());
}

//======================================================================

// The conjugate gradient solving algorithm.
vec conjugateGradientSolver( const matrix &A, const vec &B )
{
  // Setting a tolerance level which will be used as a termination condition for this algorithm
  float TOLERANCE = 1.0e-10;

  // Number of vectors/rows in the matrix A.
  int n = A.size();

  // Initializing vector X which will be set to the solution by the end of this algorithm.
  vec X( n, 0.0 );


  vec R = B;
  vec P = R;
  int k = 0;

  writeToFile(X);

  while ( k < 10 ) {
    cout << "\nIteration " << k;
    print( "\nX:", X );

    vec Rold = R;                                         // Store previous residual
    vec AP = matrixTimesVector( A, P );

    //
    float alpha = innerProduct( R, R ) / max( innerProduct( P, AP ), NEARZERO );
    X = vectorCombination( 1.0, X, alpha, P );            // Next estimate of solution
    R = vectorCombination( 1.0, R, -alpha, AP );          // Residual
    writeToFile(X);

    if ( vectorNorm( R ) < TOLERANCE ) {
      cout << "\nConverged in " << k << " iterations.\n";
      break;             // Convergence test
    }

    float beta = innerProduct( R, R ) / max( innerProduct( Rold, Rold ), NEARZERO );
    P = vectorCombination( 1.0, R, beta, P );             // Next gradient
    k++;
  }

  if(k == n) {
    cout << "\nDid not converge in " << k << " iterations.\n";
  }

  cout << endl;

  return X;
}

