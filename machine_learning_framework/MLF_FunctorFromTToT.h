#pragma once
template< class T>
class FunctorFromTToT
{
public:
	virtual T operator() (T in) = 0;
};

template< class T>
class LinearFunctorFromTToT :public FunctorFromTToT<T>
{
public:
	virtual T operator() (T in) { return in; }
};

