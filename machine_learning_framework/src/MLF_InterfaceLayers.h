#pragma once
#include <vector>

template<class T>
using VectorMatrix3D = std::vector< std::vector < std::vector<T> > >;


template <class T>
class InterfaceLayers// this is quite ugly, I just want to hold something like  vector<vector<vector<float>>>
{						//where i can do in[layer_nb][neuron_nb][weight_nb], but be able to hold just as easily 
						//queue<queue<queue<float>>>
public:
	virtual T &get(const int layer_nb, const int neuron_nb, const int weight_nb)const = 0;
	virtual int size_layers()const = 0;
	virtual int size_neurons(const int layer_nb)const = 0;
	virtual int size_weights(const int layer_nb, const int neuron_nb)const = 0;
};





template <class T>
class VectorLayers
{
public:
	VectorMatrix3D<T> a;
	virtual T &get(int layer_nb, int neuron_nb, int weight_nb)const
	{
		return a[layer_nb][neuron_nb][weight_nb];
	}
	virtual int size_layers()const
	{
		return a.size();
	}
	virtual int size_neurons(const int layer_nb)const
	{
		return a[layer_nb].size();
	}
	virtual int size_weights(const int layer_nb, const int neuron_nb)const
	{
		return a[layer_nb][neuron_nb].size();
	}
};

