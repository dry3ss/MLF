#pragma once

#include "MLF_FunctorFromTToT.h"
#include "MLF_NetworkBuilder.h"

#include <string>



template<class T, class Storage_T = VectorMatrix3D<T>>// T -> something like float
class TestNetwork
{
private:
	bool currently_working;//defautl =false, prevents any setting from happening while we are working
	FunctorFromTToT<T> *activation_function;
	NetworkStorage<Storage_T> *layers;
	//TODO: add regularization
	VectorMatrix<T> results;
public:
	TestNetwork<T, Storage_T>() :
		currently_working(false), layers(NULL), activation_function(NULL), results()
	{}
	bool setActivation_function(FunctorFromTToT<T> *f);
	bool setLayers(NetworkStorage<Storage_T> *n_l);
protected:
	void initializeResults();//initializes results to its good size
public:
	bool checkCoherentInput(int size_in); // check that number of weights in each neuron j of layer 0 = size_in =number of inputs in "in"
	NetworkStorage<Storage_T>* getLayers() {		return layers;	}
	Vector<T> getResults(Vector<T> &in);// InterfaceIO<T> -> something like vector<float>, where we can do in[index];
	std::string to_string();


};



template<class T, class Storage_T>
bool TestNetwork<T, Storage_T>::setActivation_function(FunctorFromTToT<T> *f)
{
	if (currently_working)
		return false;
	activation_function = f;
	return true;
}


template<class T, class Storage_T>
bool TestNetwork<T, Storage_T>::setLayers(NetworkStorage<Storage_T> *n_l)
{
	if (currently_working)
		return false;
	layers = n_l;
	return true;
}

template<class T, class Storage_T>
Vector<T> TestNetwork<T, Storage_T>::getResults(Vector<T> &in)
{
	if (!checkCoherentInput(in.size()) || activation_function==NULL)
		return Vector<T>();//TODO ATTENTION THIS IS TEMPORARY, should throw exception
	currently_working = true;
	initializeResults();
	for(int i = 0; i < layers->content.size() ;i++)
	{
		for (int j = 0; j < layers->content[i].size(); j ++ )
		{
			T res = 0;
			for (int k = 0; k < layers->content[i][j].size(); k++)
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

template<class T, class Storage_T>
void TestNetwork<T, Storage_T>::initializeResults()
{
	// NO NEED to check checkCoherentNetwork() or checkCoherentInput() because it has already been done beforehand
			//in getResults()
	results = VectorMatrix<T>(layers->content.size(), Vector<T>()); // layers.size() = number of layers
	for (int i = 0; i < layers->content.size() ; i++)
	{
		results[i] = Vector<T>(layers->content[i].size(), 0);// layers[i].size() = number of neurons in the layer i
	}
}


template<class T, class Storage_T>
bool TestNetwork<T, Storage_T>::checkCoherentInput(int size_in)
{
	if (layers==NULL || !layers->checkCoherentNetwork())
		return false;
	for (int j = 0; j < layers->content[0].size(); j++)
	{
		if (layers->content[0][j].size() != size_in)
			return false;
	}
	return true;
}

template<class T, class Storage_T>
std::string TestNetwork<T, Storage_T>::to_string()
{
	return std::string("Activation_function:")+((activation_function == NULL) ? "NOK" : "OK") +"\n"+ layers->to_string();
}


