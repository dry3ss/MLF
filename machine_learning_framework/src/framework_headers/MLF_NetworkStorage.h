#pragma once


#include <vector>
#include <type_traits> //for type asserts
#include <exception>
#include <cstddef> 
#include <string>

//TODO add 1 weight to each neurons to allow for affine treatment and not just linear
//TODO construct a proper iterator 

template<class T>
class NetworkStorage; //forward declaration


template<class T>
class NetworkStorageBadIterator
{
private:
	NetworkStorage<T> *source;
	size_t i, j, k;
public:
	NetworkStorageBadIterator(NetworkStorage<T> *source_=nullptr, const int i_ = 0, const int j_ = 0, const int k_ = 0)
							:i(i_), j(j_), k(k_), source(source_){}
	NetworkStorageBadIterator & operator++()
	{
		int i_m = source->content.size(), j_m = source->content[i].size(), k_m = source->content[i][j].size();
		if (k + 1 < source->content[i][j].size())
				k++;
		else
		{
			if (j + 1 < source->content[i].size())
			{
				j++;
				k = 0;
			}
			else
			{
				if (i + 1 < source->content.size())
				{
					i++;
					j = 0;
					k = 0;
				}
				else
					k++;//because end() is i = content.size() - 1, j = content[i].size() - 1, k = content[i][j].size()
			}

		}
		return *this;
	}
	NetworkStorageBadIterator operator++(int a)
	{
		NetworkStorageBadIterator result(*this);
		++(*this);
		return result;
	}
	T& operator*()
	{
		return static_cast<T&>( source->content[i][j][k]);// the exception will be directly thrown by the underlying source if need be
	}
	bool operator==(const NetworkStorageBadIterator& iter)
	{
		return (i == iter.i && j == iter.j && k == iter.k);
	}
	bool operator!=(const NetworkStorageBadIterator& iter)
	{
		return !operator==(iter);
	}
};

template<class T>
struct NetworkSchema // what's used to describe the network : 
	//number of input, array of the number of neurons per layers, number of outputs, whether or not it has one extra weight for each neuron (a constant so as to move from linear to affine)
{
	unsigned int nb_input;
	unsigned int nb_output;
	std::vector<unsigned int> neurons_per_layer;
	T min_random_weights;
	T max_random_weights;
	bool affine;//whether we have one more weight per neuron, to add an affine offset 
public:
	NetworkSchema(const unsigned int nb_input_ = 0, 
		const std::vector<unsigned int> neurons_per_layer_ = std::vector<unsigned int>(),
		 const unsigned int nb_output_ = 0, const bool affine_=true,
		  const T min_random_weights_ = -1, const T max_random_weights_ = 1)		
		:nb_input(nb_input_), nb_output(nb_output_), neurons_per_layer(neurons_per_layer_),
		 affine(affine_), min_random_weights(min_random_weights_), 
		 max_random_weights(max_random_weights_){}
	unsigned int getTotalNumberNeurons()
	{
		unsigned int nb = nb_output;
		for (size_t i = 0; i < neurons_per_layer.size(); i++)
		{
			nb_output += neurons_per_layer[i];
		}
		return nb;
	}
	std::string toStringForExport() const
	{
		std::string r="["+std::to_string(min_random_weights)+";"
			+std::to_string(max_random_weights)+"]:";
		r.append(std::to_string(nb_input)+";");
		for(unsigned int i=0;i<neurons_per_layer.size();i++)
		{
			if(i!=0)
				r.append(",");
			r.append(std::to_string(neurons_per_layer[i]));			
		}
		r.append(";");
		r.append(std::to_string(nb_output));
		r.append(";");
		r.append(affine ? "Y" : "N");
		return r;
	}
	std::string toStringVerbose() const
	{
		std::string r="#####\n#Range: ["+std::to_string(min_random_weights)+";"	+std::to_string(max_random_weights)+"]\n";
		r.append("#Inputs: "+std::to_string(nb_input));
		r.append("\n#Nodes: ");
		for(unsigned int i=0;i<neurons_per_layer.size();i++)
		{
			if(i!=0)
				r.append(",");
			r.append(std::to_string(neurons_per_layer[i]));			
		}
		r.append("\n#Outputs: ");
		r.append(std::to_string(nb_output));
		r.append("\n#Affine: ");
		r.append(affine? "Y": "N");
		r.append("\n#####\n");
		return r;
	}
};

template<class T>
class NetworkStorage
{
	using Matrix3D = std::vector< std::vector < std::vector<T> > >;
public:
	typedef NetworkStorageBadIterator<T> iterator;
	Matrix3D content;
public:

	bool checkCoherentNetwork()const; // check that it's a coherent network 
	// <=> IF it is NOT an AFFINE network then number of weights in each neuron j of layer i+1 = number of neurons in layer i
	// <=> IF IS not an AFFINE network then number of weights in each neuron j of layer i+1 = number of neurons in layer i+1
	std::string toString()const;
	NetworkStorageBadIterator<T> begin(){ return NetworkStorageBadIterator<T>(this); }
	NetworkStorageBadIterator<T> end()
	{ 
		int i = content.size() - 1, j = content[i].size() - 1, k = content[i][j].size(); 
		return NetworkStorageBadIterator<T>(this,i,j,k); 
	}
	
	inline bool checkCoherentNetworkIfAffine()const;
	inline bool checkCoherentNetworkINotfAffine()const;

	bool isAffineNetwork() const
	{
		return checkCoherentNetworkIfAffine();//it will actually only return yes if it both affine and coherent,
		//but if it's not coherent it's not really affine either ...
	}
	NetworkSchema<T> getPartialSchema()//doest not contain the range
	{
		bool is_affine_network= isAffineNetwork();
		int nb_input=0, nb_output = 0;
		//lets check if its affine or not

		const int nb_layers = content.size() - 1;//-1 for the output
		nb_input = content[0][0].size()- (is_affine_network ? 1:0); //the number of input is the number of weights in the first layer, and -1 if it's affine
		std::vector<int>neurons_per_layer = std::vector<int>(nb_layers, 0);
		for (int i = 1; i < nb_layers+1; i++)
			neurons_per_layer[i - 1] = content[i][0].size() - (is_affine_network ? 1 : 0);
		nb_output = content[nb_layers].size() - (is_affine_network ? 1 : 0);
		return 	NetworkSchema<T>(nb_input, neurons_per_layer, nb_output);
	}
};


template<class T>
// <=> IF IS an AFFINE network then number of weights in each neuron j of layer i+1 = number of neurons in layer i+1
inline bool NetworkStorage<T>::checkCoherentNetworkIfAffine()const
{
	if (content.size() <= 0)
		return false;
	for (size_t i = 1; i < content.size(); i++)
	{
		for (size_t j = 0; j < content[i].size(); j++)
		{
			if (content[i][j].size() != content[i - 1].size() + 1)
				return false;
		}
	}
	return true;
}
template<class T>
//<= > IF it is NOT an AFFINE network then number of weights in each neuron j of layer i + 1 = number of neurons in layer i
inline bool NetworkStorage<T>::checkCoherentNetworkINotfAffine()const
{
	if (content.size() <= 0)
		return false;
	for (size_t i = 1; i < content.size(); i++)
	{
		for (size_t j = 0; j < content[i].size(); j++)
		{
			if (content[i][j].size() != content[i - 1].size())
				return false;
		}
	}
	return true;
}


template<class T>
bool NetworkStorage<T>::checkCoherentNetwork()const
{
	// check that it's a coherent network 
	// <=> IF it is NOT an AFFINE network then number of weights in each neuron j of layer i+1 = number of neurons in layer i
	// <=> IF IS an AFFINE network then number of weights in each neuron j of layer i+1 = number of neurons in layer i+1
	if (content.size() <= 0)
		return false;
	bool r = checkCoherentNetworkINotfAffine();
	if (!r)
		return checkCoherentNetworkIfAffine();
	else
		return true;
}

template<class T>
std::string NetworkStorage<T>::toString()const
{
	std::string res = "Number of layers: " + std::to_string(content.size())
		+ "\nStatus: "
		+ (checkCoherentNetwork() ? "OK" : "NOK");
	for (size_t i = 0; i < content.size(); i++)
	{
		res += "\nLayer " + std::to_string(i) + " size: " + std::to_string(content[i].size());
		for (size_t j = 0; j < content[i].size(); j++)
		{
			res += "\n### Neuron " + std::to_string(j + 1) + "/" + std::to_string(content[i].size()) + " size: " + std::to_string(content[i][j].size());
			res += "\n###### ";
			for (size_t k = 0; k < content[i][j].size(); k++)
			{
				res += std::to_string(content[i][j][k]) + "  ";
			}
		}
	}
	return res;
}




