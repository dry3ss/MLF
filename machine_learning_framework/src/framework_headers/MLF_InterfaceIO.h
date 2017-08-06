
#pragma once

#include <vector>

template<class T>
class InterfaceIO // this is quite ugly, I just want to hold something like  vector<float>
{//where i can do in[index], but be able to hold just as easily 
//queue<float>
public:
	virtual T& operator[] (T x) = 0;
	virtual int size()const = 0;
};


template<class T>
class VectorIO : public InterfaceIO<T>
{
public:
	std::vector<T> a;
	virtual T& operator[] (T x)
	{
		return a[x];
	}
	virtual int size()const
	{
		return a.size();
	}
};
