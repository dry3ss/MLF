#pragma once


#include "MLF_NetworkStorage.h"

//TODO this only allow the standard type of vector, not the one usign a non-standard allocator


template<class T>
using VectorMatrix3D = std::vector< std::vector < std::vector<T> > >;

template<class T>
using VectorMatrix = std::vector < std::vector<T> >;

template<class T>
using Vector = std::vector<T>;




template<class T, class Storage_T= VectorMatrix3D<T>>
class NetworkBuilder
{
	NetworkStorage<Storage_T> network;
public:
	NetworkBuilder initialize(int input_size,Vector<int> network_sizes, int output_size);
	NetworkStorage<Storage_T> construct() { return network; }
	NetworkBuilder() :network(){}
	NetworkStorage<Storage_T> &access() { return network; } //the difference with construct is that here you are manipulating directly everything
};

/*  input_size is the number of inputs
	network_sizes is a vector containing the number of neurons to create per layer (excluding the output layer)
	output_size is the number of outputs
*/
template<class T, class Storage_T>
NetworkBuilder<T, Storage_T> NetworkBuilder<T, Storage_T>::initialize(int input_size, Vector<int> network_sizes, int output_size)
{
	network.content = VectorMatrix3D<T>(network_sizes.size()+1, VectorMatrix<T>());// +1 for output
	for (int i = 0; i < network_sizes.size(); i++)
	{
		if (i == 0)
			network.content[i] = VectorMatrix<T>(network_sizes[i], Vector<T>(input_size,0));

		else
			network.content[i] = VectorMatrix<T>(network_sizes[i], Vector<T>(network_sizes[i - 1], 0));
	}
	network.content[network_sizes.size()] = VectorMatrix<T>(output_size, Vector<T>(network_sizes[network_sizes.size() -1], 0));
	return *this;
}