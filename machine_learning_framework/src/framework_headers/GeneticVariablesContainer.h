#pragma once
#include "MLF_NetworkStorage.h"
#include "MLF_NetworkBuilder.h"


/*#######################################Container for our individuals##########################################*/

template <class T, class Type_Fitness = T>
struct IndividualContainer // individual, fitness, ranking and status
{
public:
	NetworkStorage<T> individual;
	Type_Fitness fitness;
	size_t rank; // >=0 => actual rank (yes it starts at 0)
	Vector<T> output;//the latest output of its network based on the given inputs
public:
	IndividualContainer() :individual(), fitness(0), rank(-1) {};//-1 for a size_t is the definition of std::npos for example in strings
	IndividualContainer(NetworkStorage<T> &individual_) :individual(individual_), fitness(0), rank(-1) {};

	inline std::string to_stringFitnessRank()
	{
		return std::to_string(fitness) + "::" + std::to_string(rank);
	}
};

template<class T, class Type_Fitness> using IndividualsVector = Vector < IndividualContainer<T, Type_Fitness> >;


//This one contains variables used in a lot of places,the container itself is passed as a reference
//and if the user wants to inherit this class and add his own, he can !
template
<
	class T, class Type_Fitness = T
>
struct GeneticVariablesContainer
{
	IndividualsVector<T, Type_Fitness> individuals_container;//the true individuals, with fitness & rank

	TestNetwork<T> tester;//the "tester" will give us the output of each individual NN inside our population given an input


						  //for the mating process :
	Vector<NetworkStorage<T>> indiv_copy; // a copy of the network structure of each individuals, this is necessary
										  //because the breeding process will need access to the new pop and a copy of the old population at the same time
	size_t index_out;//during the breeding process, we'll need to know which 'exit' individual will be the child of the parents
	std::pair<size_t, size_t> partners;//during the breeding process, holds the indexes of the parents

									   //this is just for convenient access outside :
	Type_Fitness current_max_fit;
	Type_Fitness current_min_fit;

	GeneticVariablesContainer()
		:individuals_container(), indiv_copy(), index_out(0), partners(0, 0), current_max_fit(0), current_min_fit(0) {}

	inline void swapIndividualsFrom(Vector<NetworkStorage<T>> &to_swap)
	{
		size_t sz = std::max(to_swap.size(), individuals_container.size());
		if (to_swap.size() < sz)
			to_swap.resize(sz);
		if (individuals_container.size() < sz)
			individuals_container.resize(sz);
		for (size_t i = 0; i < sz; i++)
		{
			individuals_container[i].individual.content.swap(to_swap[i].content);
		}
	}

	inline void initAndGenerateNewPop(const unsigned int size_pop, const NetworkSchema<T> schema)
	{

		individuals_container = IndividualsVector<T, Type_Fitness>(size_pop, IndividualContainer<T, Type_Fitness>());
		NetworkBuilder<T> builder;
		builder.initializeZeroes(schema);
		NetworkStorage<T> model = builder.construct();
		for (size_t i = 0; i < size_pop; i++)
		{
			individuals_container[i].individual = model;//first we get the right structure (filled with zeroes, but it's about the structure)
			builder.fillAnyNetwokRandom(individuals_container[i].individual, schema.min_random_weights, schema.max_random_weights);	// we fill randomly the structure		
		}

		indiv_copy = Vector<NetworkStorage<T>>(individuals_container.size(), model);// now we have a vector of Networks that all have the right size
	}

	inline void initAfterCopyOrSwapFromAnotherPop(const NetworkSchema<T> schema)//if you copied your individuals_container from somewhere else, you still need to initialize the copy to the right size
	{
		NetworkBuilder<T> builder;
		builder.initializeZeroes(schema);
		NetworkStorage<T> model = builder.construct();
		indiv_copy = Vector<NetworkStorage<T>>(individuals_container.size(), model);// now we have a vector of Networks that all have the right size
	}
	inline std::string to_stringFitnessRank()
	{
		size_t sz = individuals_container.size();
		std::string ret;
		for (size_t i = 0; i < sz; i++)
		{
			ret += std::to_string(i + 1) + "/" + std::to_string(sz) + "->"
				+ individuals_container[i].to_stringFitnessRank() + "\n";
		}
		return ret;
	}

	inline IndividualContainer<T, Type_Fitness>& getIndividualWithRank(size_t rank_ = 0)//by default we get the best one
	{
		for (size_t i = 0; i < individuals_container.size(); i++)
			if (individuals_container[i].rank == rank_)
				return individuals_container[i];
		return IndividualContainer<T, Type_Fitness>();
	}

	inline bool setTesterActivation_function(FunctorFromTToT<T> *f)
	{
		return tester.setActivation_function(f);
	}
};