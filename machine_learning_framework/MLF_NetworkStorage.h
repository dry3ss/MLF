#pragma once


#include <vector>

template<class T>
class NetworkStorage
{
public:
	T content;
public:
	bool checkCoherentNetwork(); // check that number of weights in each neuron j of layer i+1 = number of neurons in layer i
	std::string to_string();
};

template<class T>
bool NetworkStorage<T>::checkCoherentNetwork()
{
	// check that it's a coherent network 
	// <=> number of weights in each neuron j of layer i+1 = number of neurons in layer i
	if (content.size() <= 0)
		return false;
	for (int i = 1; i < content.size(); i++)
	{
		for (int j = 0; j < content[i].size(); j++)
		{
			if (content[i][j].size() != content[i - 1].size())
				return false;
		}
	}
	return true;
}

template<class T>
std::string NetworkStorage<T>::to_string()
{
	std::string res = "Number of layers: " + std::to_string(content.size())
		+ "\nStatus: "
		+ (checkCoherentNetwork() ? "OK" : "NOK");
	for (int i = 0; i < content.size(); i++)
	{
		res += "\nLayer " + std::to_string(i) + " size: " + std::to_string(content[i].size());
		for (int j = 0; j < content[i].size(); j++)
		{
			res += "\n### Neuron " + std::to_string(j + 1) + "/" + std::to_string(content[i].size()) + " size: " + std::to_string(content[i][j].size());
			res += "\n###### ";
			for (int k = 0; k < content[i][j].size(); k++)
			{
				res += std::to_string(content[i][j][k]) + "  ";
			}
		}
	}
	return res;
}




