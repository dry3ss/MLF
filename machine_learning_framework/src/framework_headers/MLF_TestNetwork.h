#pragma once

#include "MLF_FunctorFromTToT.h"
#include "MLF_NetworkBuilder.h"

#include <string>

template<class T>
using VectorMatrix3D = std::vector< std::vector < std::vector<T> > >;

template<class T>
using VectorMatrix = std::vector < std::vector<T> >;

template<class T>
using Vector = std::vector<T>;

template<class T>// T -> something like float
class TestNetwork
{

	using Storage_T = VectorMatrix3D < T > ;
private:
	bool currently_working;//defautl =false, prevents any setting from happening while we are working
	FunctorFromTToT<T> *activation_function;
	NetworkStorage<T> *layers;
	//TODO: add regularization
	VectorMatrix<T> results;
public:
	TestNetwork<T>() :
		currently_working(false), layers(nullptr), activation_function(nullptr), results()
	{}
	bool setActivation_function(FunctorFromTToT<T> *f)
	{
		if (currently_working)
			return false;
		activation_function = f;
		return true;
	}
	bool setLayers(NetworkStorage<T> *n_l)
	{
		if (currently_working)
			return false;
		layers = n_l;
		return true;
	}
protected:
	void initializeResults();//initializes results to its good size
public:
	bool checkCoherentInput(int size_in); // check that number of weights in each neuron j of layer 0 = size_in =number of inputs in "in"
	void* getLayers() { return layers; }
	void* getActivation_function() { return activation_function; }
	Vector<T> getResults(const Vector<T> &in);
	std::string toString();


};

template<class T >
Vector<T> TestNetwork<T>::getResults(const Vector<T> &in)
{
	if (!checkCoherentInput(in.size()) || activation_function==NULL)
		return Vector<T>();//TODO ATTENTION THIS IS TEMPORARY, should throw exception
	currently_working = true;
	initializeResults();
	for (size_t i = 0; i < layers->content.size(); i++)
	{
		for (size_t j = 0; j < layers->content[i].size(); j++)
		{
			T res = 0;
			for (size_t k = 0; k < layers->content[i][j].size(); k++)
			{
				if (i == 0)// if we are on the first layer, we need to get everything from the input, not results
				{
					res += layers->content[i][j][k] * in[k];
				}
				else
				{
					res += layers->content[i][j][k] * results[i - 1][k];
				}
			}
			results[i][j] = (*activation_function)(res);
		}
	}
	currently_working = false;
	return results[results.size() - 1];
}

template<class T>
void TestNetwork<T>::initializeResults()
{
	// NO NEED to check checkCoherentNetwork() or checkCoherentInput() because it has already been done beforehand
			//in getResults()
	results = VectorMatrix<T>(layers->content.size(), Vector<T>()); // layers.size() = number of layers
	for (size_t i = 0; i < layers->content.size(); i++)
	{
		results[i] = Vector<T>(layers->content[i].size(), 0);// layers[i].size() = number of neurons in the layer i
	}
}


template<class T>
bool TestNetwork<T>::checkCoherentInput(int size_in)
{
	if (layers==NULL || !layers->checkCoherentNetwork())
		return false;
	for (size_t j = 0; j < layers->content[0].size(); j++)
	{
		if (layers->content[0][j].size() != size_in)
			return false;
	}
	return true;
}

template<class T>
std::string TestNetwork<T>::toString()
{
	std::string r = std::string("TestNetwork Status:\nActivation_function:")+((activation_function == NULL) ? "NOK" : "OK") +"\n";
	r += "Network set ?:";
	r += ((layers == NULL) ? "NOK" : "OK");
	r +="\n";
	return r;
}


