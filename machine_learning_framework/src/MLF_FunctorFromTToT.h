#pragma once

#include <math.h>
#include <type_traits> //for type asserts


#define PI_VALUE 3.1415926535897932384626433832795028841971693993751058209

template< class T>
class FunctorFromTToT
{
public:
	virtual T operator() (const T in) = 0;
};

template< class T>
class LinearFunctorFromTToT :public FunctorFromTToT<T>
{
public:
	virtual T operator() (const T in) { return in; }
};

template< class T>
class AffineFunctorFromTToT :public FunctorFromTToT<T>
{
	const T a, b;
public:
	AffineFunctorFromTToT(const T a_ = 1, const T b_ = 0) :a(a_), b(b_){}
	virtual T operator() (const T in) { return a*in + b; }
};


template< class T> 
class SigmoidFunctorFromTToT :public FunctorFromTToT<T>
{
	const T a;
	const T b;
	static_assert(std::is_floating_point<T>::value, "This sigmoid implementation was not meant to work with non-floating point types"); //technically a HUUGE (like int's max value) c value might make it useful, but just use your own kind of functor
public:
	SigmoidFunctorFromTToT(const T c = 1) :a(2 * c), b(-c){} //a and b values so that with c=1 range is ]-1,1[
	
	SigmoidFunctorFromTToT(const T a_, const T b_) :a(a_),b(b_){}
	virtual T operator() (const T in)  { return b+a / (1 + std::exp( - (in) ) ); }
};


template< class T>
class AtanFunctorFromTToT :public FunctorFromTToT<T>
{
	const T a;
	const T b;
	static_assert(std::is_floating_point<T>::value, "This arctan implementation was not meant to work with non-floating point types"); //technically a HUUGE (like int's max value) c value might make it useful, but just use your own kind of functor
public:
	AtanFunctorFromTToT(const T c = 1) : a(c * 2 / static_cast<T>(PI_VALUE)), b(0){} //a and b values so that with c=1 range is ]-1,1[
	AtanFunctorFromTToT(const T a_, const T b_) : a(a_), b(b_){}
	virtual T operator() (const T in)  { return a * std::atan(in) + b; }
};




