#pragma once

#include <type_traits> //for type asserts
#include <algorithm>  //sort
#include "MLF_NetworkStorage.h"
#include "MLF_NetworkBuilder.h"

#include <utility>// pair (Breeding Selector) & Genetic breeder

#include <cmath>//for max() (Breeding Selector)



/*#######################################Container for our individuals##########################################*/

template <class T, class Type_Fitness=T>
struct IndividualContainer // individual, fitness, ranking and status
{
public:
	NetworkStorage<T> individual;
	Type_Fitness fitness;
	unsigned int rank; // >=0 => actual rank (yes it starts at 0), -1= not calculated
public:
	IndividualContainer() :individual(), fitness(0), rank(-1){};
	IndividualContainer(NetworkStorage<T> &individual_) :individual(individual_), fitness(0), rank(-1){};

	inline std::string to_stringFitnessRank()
	{
		return std::to_string(fitness) + "::" + std::to_string(rank);
	}
};


/*#######################################Containers for parameters##########################################*/
template<class T, class Type_Fitness> using IndividualsVector = std::vector < IndividualContainer<T, Type_Fitness> >;


//This first one contains all the functions that the user can choose to change, for each of them, an operator() must exist! 
//Also the whole point of having  GeneticParametersContainer afterwards is that none of their constructors should take any parameters, or at least a defautl one should exist
template
<
	class T,
	class SortingIndexComparisonFunctionType,
	class BreedingSelectionFunctionType,
	class BreedingMatingFunctionType,
	class MutationFunctionType,
	class Type_Fitness=T
>
struct GeneticFunctionsContainer
{
	//static_assert(std::is_same<     decltype(BreedingSelectionFunctionType::func()), void>::value, "BreedingSelectionFunctionType must have a \"func()\" function");
	SortingIndexComparisonFunctionType sorting_index_comparison_function_type;//ATTENTIOn this one works differently, its constructor is called with the GeneticContainers, and the operator() looks like this: operator() (size_t lhs, size_t  rhs) 
	BreedingSelectionFunctionType breeding_selection_function; 
	BreedingMatingFunctionType breeding_mating_function;
	MutationFunctionType mutation_function;
	GeneticFunctionsContainer() :
		sorting_index_comparison_function_type(), breeding_selection_function(), breeding_mating_function(), mutation_function(){}
	std::string to_string()
	{
		return std::string("breeding_mating_function::") + typeid(breeding_mating_function).name()
			+ "\nbreeding_selection_function::" + typeid(breeding_selection_function).name()
			+ "\nsorting_index_comparison_function_type::" + typeid(sorting_index_comparison_function_type).name()
			+ "\nmutation_function::" + typeid(mutation_function).name();
	}
};



//This second one contains constant parameters, the container itself is passed as a reference to constant, 
//not necessarilly useful for most functions, but if the user wants to inherit this class and add his own, he can !

template
<
	class T, class Type_Fitness = T
>
struct GeneticParametersContainer
{
	NetworkAutoConstructorParameters<T> auto_constructor_params;
	//for mating:
	const float mating_proportion1_over2;
	float const_mutation_rate;
	//for mutation

	GeneticParametersContainer(const NetworkAutoConstructorParameters<T> &auto_constructor_params_, const float const_mutation_rate_percents = 1, const float prop_1_over_2 = 0.5)
		:auto_constructor_params(auto_constructor_params_),mating_proportion1_over2(prop_1_over_2)
	{
		const_mutation_rate = const_mutation_rate_percents*auto_constructor_params.schema.getTotalNumberNeurons() / 100.0;
	}
};

//This third one contains variables used in a lot of places,the container itself is passed as a reference
//and if the user wants to inherit this class and add his own, he can !
template
<
	class T, class Type_Fitness=T
>
struct GeneticVariablesContainer
{
	IndividualsVector<T, Type_Fitness> individuals_container;//the true individuals, with fitness & rank
	//for mating:
	std::vector<NetworkStorage<T>> indiv_copy; // a copy of the network structure of each individuals
	size_t index_out;
	std::pair<size_t, size_t> partners;

	Type_Fitness current_max_fit;
	Type_Fitness current_min_fit;

	GeneticVariablesContainer()
		:individuals_container(), indiv_copy(), index_out(0), partners(0, 0), current_max_fit(0), current_min_fit(0) {}

	inline void swapIndividualsFrom(std::vector<NetworkStorage<T>> &to_swap)
	{
		size_t sz = std::max(to_swap.size(), individuals_container.size());
		for (size_t i = 0; i < sz; i++)
		{
			individuals_container[i].individual.content.swap(to_swap[i].content);
		}
		initCopy();
	}

	inline void init(const unsigned int size_pop, const NetworkAutoConstructorParameters<T> param)
	{

		individuals_container = IndividualsVector<T, Type_Fitness>(size_pop, IndividualContainer<T, Type_Fitness>());
		NetworkBuilder<T> builder;
		builder.initializeZeroes(param.schema);
		NetworkStorage<T> model = builder.construct();
		for (size_t i = 0; i < size_pop; i++)
		{
			individuals_container[i].individual = model;//first we get the right structure (filled with zeroes, but it's about the structure)
			builder.fillAnyNetwokRandom(individuals_container[i].individual, param.min_random_weights, param.max_random_weights);	// we fill randomly the structure		
		}

		indiv_copy = std::vector<NetworkStorage<T>>(individuals_container.size(), model);// now we have a vector of Networks that all have the right size
	}

	inline void initCopy()//if you copied your individuals_container from somewhere else, you still need to initialize the copy to the right size
	{
		NetworkSchema schema = to_breed[0].individual.getSchema();
		NetworkBuilder<T> builder;
		builder.initializeZeroes(schema);
		NetworkStorage<T> model = builder.construct();
		indiv_copy = std::vector<NetworkStorage<T>>(variables_container.individuals_container.size(), model);// now we have a vector of Networks that all have the right size
	}
	inline std::string to_stringFitnessRank()
	{
		size_t sz = individuals_container.size();
		std::string ret;
		for (size_t i = 0; i < sz; i++)
		{
			 ret +=std::to_string(i + 1) + "/" + std::to_string(sz) + "->"
				+ individuals_container[i].to_stringFitnessRank()+"\n";
		}
		return ret;
	}
};



/*#######################################Custom comparison (index based) for our containers##########################################*/

template <class T, class Type_Fitness=T>
class indexComparisonInterface
{
protected:
	const std::vector<IndividualContainer<T, Type_Fitness>> *to_sort;
public:
	void setParameters(const GeneticVariablesContainer<T, Type_Fitness> &variables_container, const GeneticParametersContainer<T,Type_Fitness>  &params_container)
	{
		to_sort=&(variables_container.individuals_container);
	}
	indexComparisonInterface()
		:to_sort(nullptr){}
	virtual bool operator() (size_t lhs, size_t  rhs) = 0;
};

template <class T, class Type_Fitness=T>
class indexComparisonStd : public indexComparisonInterface<T, Type_Fitness>
{
public:
	bool operator() (size_t lhs, size_t rhs) { return (*to_sort)[lhs].fitness> (*to_sort)[rhs].fitness; }
};


/*#######################################Breeding Selector##########################################*/
template <class T, class Type_Fitness >
class GeneticBreederInterface; //forward declaration


template <class T, class Type_Fitness=T>
class GeneticBreedSelectorInterface
{
public:
	virtual inline void operator()(GeneticVariablesContainer<T, Type_Fitness>&variables_container, const GeneticParametersContainer<T,Type_Fitness>  &params_container) = 0;
};

/* ATTENTION: this class will not work (at least it will be as if the negative values are 0) if you use negative fitness as well as positive !
	Also note: Fitness is assumed to be higher=better //TODO make that behaviour configurable
*/
template <class T, class Type_Fitness=T>
class GeneticBreedFitnessBasedSelector : public GeneticBreedSelectorInterface<T, Type_Fitness>
{
protected:
	std::vector<Type_Fitness> roulettes_vec;
protected:

	inline size_t getIndexSelectedIndividual(Type_Fitness &rand_)
	{
		Type_Fitness curr_sum = 0;
		//TODO check if the defautl param works !
		for (size_t i = 0; i < roulettes_vec.size(); i++)
		{
			curr_sum += roulettes_vec[i];
			if (rand_ <= curr_sum)
				return i;
		}
		return roulettes_vec.size();
	}
	inline size_t getIndexSelectedIndividual(Type_Fitness &rand_, size_t to_skip )
	{
		Type_Fitness curr_sum = 0;
		//TODO check if the defautl param works !
		for (size_t i = 0; i < roulettes_vec.size(); i++)
		{
			if (i == to_skip)
				continue;//we do wanna skip him !
			curr_sum += roulettes_vec[i];
			if (rand_ <= curr_sum)
				return i;
		}
		return roulettes_vec.size();
	}
public:
	GeneticBreedFitnessBasedSelector() :roulettes_vec(){}
	virtual inline void operator()(GeneticVariablesContainer<T, Type_Fitness>&variables_container, const GeneticParametersContainer<T,Type_Fitness>  &params_container)
	{
		IndividualsVector<T, Type_Fitness> &to_breed=variables_container.individuals_container;
		//let's define a vector containing the "border values" of our roulette
		roulettes_vec = std::vector<Type_Fitness>(to_breed.size(), 0);
		Type_Fitness sum = 0;
		//let's populate our roulette
		for (size_t i = 0; i < to_breed.size(); i++)
		{
			Type_Fitness current_fit = std::max(static_cast<Type_Fitness>(0), (to_breed[i].fitness));// max in case we used negative values
			sum += current_fit;
			roulettes_vec[i] = current_fit;
		}		
		// let's set up our random generator with its distribution
		auto uniform_distrib = RandomManagement::RandomEngineGlobal.getUniformDistribution(static_cast<Type_Fitness>(0), sum); //uniform between 0 and our sum ( true =>sum included if we are dealing with floating points random gen)

		//get our random result:
		Type_Fitness rand_ = uniform_distrib(RandomManagement::RandomEngineGlobal.getMT());
		//find which individual was selected with our inline function
		size_t indiv1=getIndexSelectedIndividual(rand_);
		if (indiv1 >= roulettes_vec.size())
			throw std::out_of_range("Did not find our individual 1 in GeneticBreedFitnessBasedSelector::select_two_partners()");
		//once again for the second one, but we have to remove our first one beforehand !
		//TODO debug only:
		Type_Fitness sum_ = sum - roulettes_vec[indiv1];
		uniform_distrib = RandomManagement::RandomEngineGlobal.getUniformDistribution(static_cast<Type_Fitness>(0), sum - roulettes_vec[indiv1]);
		rand_ = uniform_distrib(RandomManagement::RandomEngineGlobal.getMT());
		size_t indiv2 = getIndexSelectedIndividual(rand_,indiv1);
		if (indiv2 >= roulettes_vec.size())
			throw std::out_of_range("Did not find our individual 2 in GeneticBreedFitnessBasedSelector::select_two_partners()");
		variables_container.partners=std::pair<size_t, size_t>(indiv1, indiv2);
	}
};
/*#######################################Breeding: Mating##########################################*/
template <class T, class Type_Fitness=T>
class GeneticBreedMatingInterface
{
public:
	virtual inline void operator()(GeneticVariablesContainer<T, Type_Fitness>&variables_container, const GeneticParametersContainer<T,Type_Fitness>  &params_container) = 0;
};

template <class T, class Type_Fitness=T>
class GeneticBreedUniformMating : public GeneticBreedMatingInterface<T, Type_Fitness>
{
public:
	/*ATTENTION this function will only work if all the sizes are correctly configured to be the same*/
	virtual inline void operator()(GeneticVariablesContainer<T, Type_Fitness>&variables_container, const GeneticParametersContainer<T,Type_Fitness>  &params_container)
	{

		const size_t &index_out = variables_container.index_out;
		const std::pair<size_t, size_t> &partners = variables_container.partners;
		std::vector<NetworkStorage<T>> &indiv_copy = variables_container.indiv_copy;
		const float &proportion1_over2 = params_container.mating_proportion1_over2;

		IndividualsVector<T, Type_Fitness> &out_container = variables_container.individuals_container;


		NetworkStorage<T>::iterator out_iter = out_container[index_out].individual.begin();
		NetworkStorage<T>::iterator partner1_iter = indiv_copy[partners.first].begin();
		NetworkStorage<T>::iterator partner2_iter = indiv_copy[partners.second].begin();
		auto uniform_distrib = RandomManagement::RandomEngineGlobal.getUniformDistribution(0.0, 1.0);
		while (out_iter != out_container[index_out].individual.end())
		{
			double rand_ = uniform_distrib(RandomManagement::RandomEngineGlobal.getMT());
			if (rand_ < proportion1_over2)
				*out_iter = *partner1_iter;
			else
				*out_iter = *partner2_iter;
			out_iter++; partner1_iter++; partner2_iter++;
		}
	
	}
};


/*#######################################Mutation##########################################*/
template <class T, class Type_Fitness = T>
class GeneticMutationInterface
{
public:
	virtual inline void operator()(GeneticVariablesContainer<T, Type_Fitness>&variables_container, const GeneticParametersContainer<T,Type_Fitness>  &params_container) = 0;
};

template <class T, class Type_Fitness = T>
class GeneticMutationConst: GeneticMutationInterface<T, Type_Fitness>
{
public:
	virtual inline void operator()(GeneticVariablesContainer<T, Type_Fitness>&variables_container, const GeneticParametersContainer<T,Type_Fitness>  &params_container)
	{
		const float mut_rate = params_container.const_mutation_rate;
		size_t sz = variables_container.individuals_container.size();
		//btw 0 and 1 to choose if we are indeed going to mutate
		auto distrib_selection = RandomManagement::RandomEngineGlobal.getUniformDistribution(0.0, 1.0);
		//this one has the same range we used to construct our neuron's weights
		auto distrib_weights = RandomManagement::RandomEngineGlobal.getUniformDistribution(params_container.auto_constructor_params.min_random_weights, params_container.auto_constructor_params.max_random_weights);
		for (size_t a = 0; a < sz; a++)
		{
			NetworkStorage<T> current_indiv=variables_container.individuals_container[a].individual;
			for (NetworkStorage<T>::iterator i = current_indiv.begin(); i != current_indiv.end(); i++)
			{
				double choice = distrib_selection(RandomManagement::RandomEngineGlobal.getMT());
				if (choice <= mut_rate)
				{
					*i=distrib_weights(RandomManagement::RandomEngineGlobal.getMT());
				}
			}
		}
	}
};

//here the rate of mutation depends on the rank of the previous individual that was in your spot in the last generation, the more well ranked he was, the less mutations you should have
template <class T, class Type_Fitness = T>
class GeneticMutationSimpleRank : GeneticMutationInterface<T, Type_Fitness>
{
public:
	virtual inline void operator()(GeneticVariablesContainer<T, Type_Fitness>&variables_container, const GeneticParametersContainer<T, Type_Fitness>  &params_container)
	{
		const float max_mut_rate = params_container.const_mutation_rate;
		size_t sz = variables_container.individuals_container.size();
		//btw 0 and 1 to choose if we are indeed going to mutate
		auto distrib_selection = RandomManagement::RandomEngineGlobal.getUniformDistribution(0.0, 1.0);
		//this one has the same range we used to construct our neuron's weights
		auto distrib_weights = RandomManagement::RandomEngineGlobal.getUniformDistribution(params_container.auto_constructor_params.min_random_weights, params_container.auto_constructor_params.max_random_weights);
		for (size_t a = 0; a < sz; a++)
		{
			NetworkStorage<T> current_indiv = variables_container.individuals_container[a].individual;
			unsigned int current_rank = variables_container.individuals_container[a].rank;
			float current_mut_rate = current_rank / static_cast<float>(sz)*max_mut_rate;
			for (NetworkStorage<T>::iterator i = current_indiv.begin(); i != current_indiv.end(); i++)
			{
				double choice = distrib_selection(RandomManagement::RandomEngineGlobal.getMT());
				if (choice <= current_mut_rate)
				{
					*i = distrib_weights(RandomManagement::RandomEngineGlobal.getMT());
				}
			}
		}
	}
};

//here the values of mutation as well as the rate, both depend on the rank of the previous individual that was in your spot in the last generation, the more well ranked he was, the less mutations you should haveand the less profound they'll be
template <class T, class Type_Fitness = T>
class GeneticMutationDoubleRank : GeneticMutationInterface<T, Type_Fitness>
{
public:
	virtual inline void operator()(GeneticVariablesContainer<T, Type_Fitness>&variables_container, const GeneticParametersContainer<T, Type_Fitness>  &params_container)
	{
		const float max_mut_rate = params_container.const_mutation_rate;
		size_t sz = variables_container.individuals_container.size();
		auto distrib_selection = RandomManagement::RandomEngineGlobal.getUniformDistribution(0.0, 1.0);
		Type_Fitness random_range = params_container.auto_constructor_params.max_random_weights -params_container.auto_constructor_params.min_random_weights;
		auto distrib_weights = RandomManagement::RandomEngineGlobal.getUniformDistribution(params_container.auto_constructor_params.min_random_weights, params_container.auto_constructor_params.max_random_weights);
		for (size_t a = 0; a < sz; a++)
		{
			unsigned int current_rank = variables_container.individuals_container[a].rank;
			if (current_rank == 0)
				continue;
			NetworkStorage<T> current_indiv = variables_container.individuals_container[a].individual;
			float current_mut = current_rank / static_cast<float>(sz);
			Type_Fitness current_range = random_range*current_mut;//no "overflow" from our range this way
			for (NetworkStorage<T>::iterator i = current_indiv.begin(); i != current_indiv.end(); i++)
			{
				double choice = distrib_selection(RandomManagement::RandomEngineGlobal.getMT());
				if (choice <= (current_mut*max_mut_rate))
				{
					Type_Fitness n_a = (*i) - current_range, n_b = (*i) + current_range;
					Type_Fitness n_range = n_b - n_a;
					*i = n_range*distrib_selection(RandomManagement::RandomEngineGlobal.getMT()) + n_a;
				}
			}
		}
	}
};

/*#######################################Genetic Optimizer##########################################*/


template
<
	class T, 
	class GeneticFunctionsContainerType,
	class Type_Fitness=T
>
class GeneticOptimizer
{
protected:
	GeneticFunctionsContainerType &function_container;
	const GeneticParametersContainer<T,Type_Fitness>  &params_container;
	GeneticVariablesContainer<T, Type_Fitness> &variables_container;


public:
	GeneticOptimizer(GeneticFunctionsContainerType &function_container_, const GeneticParametersContainer<T,Type_Fitness>  &params_container_, GeneticVariablesContainer<T, Type_Fitness>&variables_container_)
		: function_container(function_container_), params_container(params_container_), variables_container(variables_container_)
	{}

	//TODO change that for a ranking as the fitness is filled !
	inline void generateRanks()//first action in a cycle: rank every individual according to their fitness
	{
		IndividualsVector<T, Type_Fitness> &individual_containers = variables_container.individuals_container;

		function_container.sorting_index_comparison_function_type.setParameters(variables_container, params_container);

		std::vector<size_t> index_to_sort = std::vector<size_t>(individual_containers.size(), 0);//make a copy of all indices, that's what we'll sort
		for (size_t i = 0; i < individual_containers.size(); i++)
		{
			index_to_sort[i] = i;//fill the copy
		}
		std::sort(index_to_sort.begin(), index_to_sort.end(), function_container.sorting_index_comparison_function_type);//sort it with a special sorting which translates the index into the fitness and compares after
		
		for (size_t i = 0; i < individual_containers.size(); i++)
		{
			individual_containers[index_to_sort[i]].rank = i; //set the calculated ranks 
		}
		variables_container.current_max_fit= individual_containers[index_to_sort[0]].fitness;
		variables_container.current_min_fit = individual_containers[index_to_sort[index_to_sort.size()-1]].fitness;
	}

	//breed_selector =the function used for selecting mates in the mating process
	inline void breed()
	{
		//ATTENTION this is only designed for a to_breed vector of individuals having all the same Network structure( ie number of wieghts and neurons ...)

		IndividualsVector<T, Type_Fitness> &to_breed = variables_container.individuals_container;


		std::vector<NetworkStorage<T>> &indiv_copy = variables_container.indiv_copy;//this one has already been initialized by the user, by a call to variables_container.init() or .initCopy()
		for (size_t a = 0; a < to_breed.size(); a++)
		{
			//we have created a (bad for now)iterator for our NetworkStorage just for this moment!
			//for (NetworkStorage<T>::iterator i = to_breed[a].individual.begin(); i != to_breed[a].individual.end(); i++)
			indiv_copy[a].content = to_breed[a].individual.content;
		}

		//okay now we run through our individuals
		for (size_t i = 0; i < to_breed.size(); i++)
		{
			if (to_breed[i].rank == 0)//we never touch the very best one
				continue;
			variables_container.index_out = i;
			//partners selection :
			function_container.breeding_selection_function(variables_container, params_container);
			//hybridation :
			function_container.breeding_mating_function(variables_container, params_container);
		}
		/*
		//now we have to swap (to avoid copy time) our to_breed[i].individual.content and  our indiv_copy[i].content
		for (size_t i = 0; i < to_breed.size(); i++)
		{
			if (to_breed[i].rank == 1)//since it's not in indiv_copy because we didn't touch it, we have to keep it
				continue;
			to_breed[i].individual.content.swap(indiv_copy[i].content);
		}
		*/
	}
	inline void mutate()
	{
		function_container.mutation_function(variables_container, params_container);
	}
};

