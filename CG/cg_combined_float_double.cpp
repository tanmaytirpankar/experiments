#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
using namespace std;

const double NEARZERO = 1.0e-10;       // interpretation of "zero"
const float NEARZERO_f = 1.0e-10;       // interpretation of "zero"

template<class T>
using vec    = vector<T>;         // vector

template<class T>
using matrix = vector<vec<T>>;    // matrix (=collection of (row) vectors)

ofstream data_file;

// Prototypes
template<class T>
void print( string title, const vec<T> &V);

template<class T>
void print( string title, const matrix<T> &A );

template<class T>
void writeToFile(const vec<T> &V);

template<class T>
vec<T> matrixTimesVector( const matrix<T> &A, const vec<T> &V );

template<class T>
vec<T> vectorCombination( T a, const vec<T> &U, T b, const vec<T> &V );

template<class T>
T innerProduct( const vec<T> &U, const vec<T> &V );

template<class T>
T vectorNorm( const vec<T> &V );

template<class T>
T cosine_similarity(const vec<T> &U, const vec<T> &V);

vec<double> conjugateGradientSolver( const matrix<double> &A_d, const vec<double> &B_d,
                                const matrix<float> &A_f, const vec<float> &B_f);


//======================================================================


int main()
{
  matrix<double> A_d = { { 3.96751, 1.59033 }, { 0.72374, 1.59008 } };
  vec<double> B_d = { 1.19822, 0.359596 };
  matrix<float> A_f = { { 3.96751, 1.59033 }, { 0.72374, 1.59008 } };
  vec<float> B_f = { 1.19822, 0.359596 };

  data_file.open ("data_half.csv");

  cout << "Solves AX = B\n";
  print( "\nA:", A_d );
  print( "\nB:", B_d );

  vec<double> X = conjugateGradientSolver( A_d, B_d, A_f, B_f );

  cout << "\nSolution:";
  print( "\nX:", X );
  print( "\nCheck AX:", matrixTimesVector( A_d, X ) );

  data_file.close();
  return 0;
}


//======================================================================

// Prints the vector V
template<class T>
void print( string title, const vec<T> &V)
{
  cout << title << '\n';

  int n = V.size();
  for ( int i = 0; i < n; i++ )
  {
    double x = V[i];   if ( abs( x ) < NEARZERO ) x = 0.0;
    cout << x << '\t';
  }
  cout << '\n';
}


//======================================================================

// Prints the matrix A
template<class T>
void print( string title, const matrix<T> &A )
{
  cout << title << '\n';

  int m = A.size(), n = A[0].size();                      // A is an m x n matrix
  for ( int i = 0; i < m; i++ )
  {
    for ( int j = 0; j < n; j++ )
    {
      double x = A[i][j];   if ( abs( x ) < NEARZERO ) x = 0.0;
      cout << x << '\t';
    }
    cout << '\n';
  }
}


//======================================================================

// Writes the vector V to a file
template<class T>
void writeToFile(const vec<T> &V) {
  data_file << V[0];
  for (int i = 1; i < V.size(); i++) {
    data_file << "," << V[i];
  }
  data_file << endl;
}

//======================================================================

// Inner product of the matrix A with vector V returned as C (a vector)
template<class T>
vec<T> matrixTimesVector( const matrix<T> &A, const vec<T> &V )     // Matrix times vector
{
  int n = A.size();
  vec<T> C( n );
  for ( int i = 0; i < n; i++ ) C[i] = innerProduct( A[i], V );
  return C;
}


//======================================================================

// Returns the Linear combination of aU+bV as a vector W.
template<class T>
vec<T> vectorCombination( T a, const vec<T> &U, T b, const vec<T> &V )        // Linear combination of vectors
{
  int n = U.size();
  vec<T> W( n );
  for ( int j = 0; j < n; j++ ) W[j] = a * U[j] + b * V[j];
  return W;
}


//======================================================================

// Returns the inner product of vector U with V.
template<class T>
T innerProduct( const vec<T> &U, const vec<T> &V )          // Inner product of U and V
{
  return inner_product( U.begin(), U.end(), V.begin(), 0.0 );
}

double innerProduct( const vec<double> &U, const vec<float> &V )          // Inner product of U and V
{
  return inner_product( U.begin(), U.end(), V.begin(), 0.0 );
}


//======================================================================

// Computes and returns the Euclidean/2-norm of the vector V.
template<class T>
T vectorNorm( const vec<T> &V )                       // Vector norm
{
  return sqrt( innerProduct( V, V ) );
}


//======================================================================

// Computes and returns the cosine similarity of the vector U and V.
double cosine_similarity(const vec<double> &U, const vec<float> &V) {
  return innerProduct(U, V) / (vectorNorm(U) * vectorNorm(V));
}

//======================================================================

// The conjugate gradient solving algorithm.
vec<double> conjugateGradientSolver( const matrix<double> &A_d, const vec<double> &B_d,
                                     const matrix<float> &A_f, const vec<float> &B_f)
{
  // Setting a tolerance level which will be used as a termination condition for this algorithm
  double TOLERANCE_d = 1.0e-10;
  float TOLERANCE_f = 1.0e-10;

  // Number of vectors/rows in the matrix A.
  int n = A_d.size();

  // Initializing vector X which will be set to the solution by the end of this algorithm.
  vec<double> X_d( n, 0.0 );
  vec<float> X_f( n, 0.0 );


  vec<double> R_d = B_d;
  vec<float> R_f = B_f;
  vec<double> P_d = R_d;
  vec<float> P_f = R_f;
  int k = 0;

  writeToFile(X_d);

  while ( k < 10 ) {
    cout << "\nIteration " << k;
    print( "\nX_d:", X_d );
    print( "\nX_f:", X_f );

    vec<double> Rold_d = R_d;                                         // Store previous residual
    vec<float> Rold_f = R_f;
    vec<double> AP_d = matrixTimesVector( A_d, P_d );
    vec<float> AP_f = matrixTimesVector( A_f, P_f );

    //
    double alpha_d = innerProduct( R_d, R_d ) / max( innerProduct( P_d, AP_d ), NEARZERO );
    float alpha_f = innerProduct( R_f, R_f ) / max( innerProduct( P_f, AP_f ), NEARZERO_f );
    cout << "Cosine Similarity: " <<
    cosine_similarity(vectorCombination(0.0, X_d, alpha_d, P_d ),
                      vectorCombination(0.0f, X_f, alpha_f, P_f)) << endl;

    X_d = vectorCombination( 1.0, X_d, alpha_d, P_d );            // Next estimate of solution
    X_f = vectorCombination( 1.0f, X_f, alpha_f, P_f );
    R_d = vectorCombination( 1.0, R_d, -alpha_d, AP_d );          // Residual
    R_f = vectorCombination( 1.0f, R_f, -alpha_f, AP_f );
    writeToFile(X_d);

    if ( vectorNorm( R_d ) < TOLERANCE_d ) {
      cout << "\nConverged in " << k << " iterations.\n";
      break;             // Convergence test
    }

    double beta_d = innerProduct( R_d, R_d ) / max( innerProduct( Rold_d, Rold_d ), NEARZERO );
    float beta_f = innerProduct( R_f, R_f ) / max( innerProduct( Rold_d, Rold_d ), NEARZERO );
    P_d = vectorCombination( 1.0, R_d, beta_d, P_d );             // Next gradient
    P_f = vectorCombination( 1.0f, R_f, beta_f, P_f );
    k++;
  }

  if(k == n) {
    cout << "\nDid not converge in " << k << " iterations.\n";
  }

  cout << endl;

  return X_d;
}