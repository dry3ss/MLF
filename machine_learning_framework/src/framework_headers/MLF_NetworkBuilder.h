#pragma once


#include "MLF_NetworkStorage.h"
#include "RandomEngine.h"

#include <cstdlib>
#include <ctime>
#include <limits>


template <class T>
class NetworkBuilder
{

	using Matrix3D = std::vector< std::vector < std::vector<T> > >;
	using Matrix = std::vector < std::vector<T> >;
	using Vector = std::vector<T>;
	/*
	// 3 names we are really going to use, everything before was just for this:	
	using Matrix3D = V1 < V2<V3<T, A3>, A2>, A1 > ;
	using Matrix = V2 < V3<T, A3>, A2 > ;
	using Vector = V3 < T, A3 > ;
	*/
protected:
	NetworkStorage<T> network;
	NetworkSchema<T> network_schema;
public:
	/*  input_size is the number of inputs
	network_sizes is a vector containing the number of neurons to create per layer (excluding the output layer)
	output_size is the number of outputs
	*/
	void initializeZeroes(const int input_size, const std::vector<unsigned int> network_sizes, const int output_size, const bool affine_=true)
	{
		network_schema = NetworkSchema<T>(input_size, network_sizes, output_size, affine_);//update the schema
		network.content = Matrix3D(network_sizes.size() + 1, Matrix());// +1 for output
		if(!affine_)
		{ 
			for (size_t i = 0; i < network_sizes.size(); i++)
			{
				if (i == 0)
					network.content[i] = Matrix(network_sizes[i], Vector(input_size, 0));

				else
					network.content[i] = Matrix(network_sizes[i], Vector(network_sizes[i - 1], 0));
			}
			network.content[network_sizes.size()] = Matrix(output_size, Vector(network_sizes[network_sizes.size() - 1], 0));
		}
		else
		{
			for (size_t i = 0; i < network_sizes.size(); i++)
			{
				if (i == 0)
					network.content[i] = Matrix(network_sizes[i], Vector(input_size+1, 0));

				else
					network.content[i] = Matrix(network_sizes[i], Vector(network_sizes[i - 1] + 1, 0));
			}
			network.content[network_sizes.size()] = Matrix(output_size, Vector(network_sizes[network_sizes.size() - 1]+1, 0));
		}
	}
	void initializeZeroes(const NetworkSchema<T> s)
	{
		initializeZeroes(s.nb_input, s.neurons_per_layer, s.nb_output,s.affine);
	}
	NetworkStorage<T> construct() { return network; }
	NetworkBuilder() :network(), network_schema(){}
	NetworkStorage<T> &access() { return network; } //the difference with construct is that here you are manipulating directly everything , BE CAREFUL here the network_schema is not updated!

	static void fillAnyNetwokRandom(NetworkStorage<T> &network_,const T min_included, const T max)
	{
		auto distrib = RandomManagement::RandomEngineGlobal.getUniformDistribution(min_included, max);
			
		for (size_t i = 0; i < network_.content.size(); i++)
		{
			for (size_t j = 0; j < network_.content[i].size(); j++)
			{
				for (size_t k = 0; k < network_.content[i][j].size(); k++)
				{
					network_.content[i][j][k] = distrib(RandomManagement::RandomEngineGlobal.getMT());
				}
			}
		}
	}
	void fillRandom(const T min_included, const T max)
	{
		network_schema.min_random_weights = min_included;
		network_schema.max_random_weights = max;
		fillAnyNetwokRandom(network, min_included, max);
	}

	NetworkSchema<T> getNetworkSchema(){ return network_schema; }
	void autoInitializeAndFill(const NetworkSchema<T> &param)
	{
		network_schema = param;
		initializeZeroes(network_schema);
		fillRandom(network_schema.min_random_weights, network_schema.max_random_weights);
	}
};

/* Not needed anymore, kept just in case right now



template <class T>
class NetworkBuilderRandomReal : public NetworkBuilder<T>
{
private:
	std::uniform_real_distribution<T> distrib_real;
public:
	NetworkBuilderRandomReal() :distrib_real(0,1.0){	}
	 
	void fillRandomReal(const T min_included, const T max)
	{
		if (min_included != distrib_real.min() ||
			(max != distrib_real.max()) )
		{
				distrib_real = std::uniform_real_distribution<T>(min_included, max_included);
		}
		for (size_t i = 0; i < network.content.size(); i++)
		{
			for (size_t j = 0; j < network.content[i].size(); j++)
			{
				for (size_t k = 0; k < network.content[i][j].size(); k++)
				{
					network.content[i][j][k] = distrib_real(RandomManagement::RandomEngineGlobal.getMT());
				}
			}
		}
		
	}
	
};


template <class T>
class NetworkBuilderRandomInt : public NetworkBuilder<T>
{
private:
	//Note: this will not work if T is char, should use short and truncate then
	std::uniform_int_distribution<T> distrib_int;
public:
	void fillRandomInt(const T min_included, const T max_included)
	{
		if (distrib_int.min() != min_included || distrib_int.max() != max_included)
			distrib_int = std::uniform_int_distribution<T>(min_included, max_included);
		for (int i = 0; i < network.content.size(); i++)
		{
			for (int j = 0; j < network.content[i].size(); j++)
			{
				for (int k = 0; k < network.content[i][j].size(); k++)
				{
					network.content[i][j][k] = distrib_int(RandomManagement::RandomEngineGlobal.getMT());
				}
			}
		}
	}

};




*/