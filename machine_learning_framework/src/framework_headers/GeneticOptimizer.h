#pragma once

#include <type_traits> //for type asserts
#include <algorithm>  //sort

#include "GeneticFunctionContainer.h"

#include "GeneticBreeder.h"//the GeneticVariablesContainer & GeneticVariablesContainer are contained in those
#include "GeneticMutation.h"
#include "FitnessFunctionInterface.h"

#include <typeinfo> // for typeid

#include <utility>// pair (Breeding Selector) & Genetic breeder

#include <cmath>//for max() (Breeding Selector)





/*#######################################Custom comparison (index based) for our containers##########################################*/
// This is the comparator obj that will be used to rank all individuals 
//based on their fitness score (the ranking can then be used during breeding/mutation).
//This one NEEDS to be an child of the indexComparisonInterface interface class.
//It cannot be a simple comparison because the indexComparisonStd will need to compares scores 
//given only the 2 indexes of the individuals, not the scores themselves, because
//we want to rank the individuals, not only the scores and we want to be able to use std::sort().
//So the way this works is as follows:
// -before the actual ranking, the comparator object is initialized so has to get pointers to the
//GeneticParametersContainer and GeneticVariablesContainer and thus access to the array inside
//the GeneticVariablesContainer containing the scores for each individual, it stores a pointer
//to this array as this->to_sort
// -a new array (index_to_sort) containing all the integers from 0 to [number of elements in the array to sort]
//is created and this will be the one we will sort using std::sort() and passing the comparator
//object as its 3rd argument
// -the comparator  only has access to the the 2 elements in index_to_sort,and since those are indices in the
//constant (during the sorting at least) array from GeneticVariablesContainer containing the scores of the 
//individuals so it only has to do a comparison between the two elements referenced by:
//(to_sort->at(lhs).fitness) and  (to_sort->at(rhs).fitness) such as : 
//(to_sort->at(lhs).fitness) > (to_sort->at(rhs).fitness) for example for indexComparisonStd to get
//NOTE: since we are using std::sort() and a ">" comparison, here with indexComparisonStd, the higher
//the fitness, the "better" it will be
template <class T, class Type_Fitness = T>
class indexComparisonInterface
{
protected:
	const Vector<IndividualContainer<T, Type_Fitness>> *to_sort;
public:
	void setParameters(const GeneticVariablesContainer<T, Type_Fitness> &variables_container, const GeneticParametersContainer<T, Type_Fitness>  &params_container)
	{
		to_sort = &(variables_container.individuals_container);
	}
	indexComparisonInterface()
		:to_sort(nullptr) {}
	virtual bool operator() (size_t lhs, size_t  rhs) = 0;
};
//indexComparisonStd ranks in a "higher is better" fashion for the fitness: higher fitness=lower rank
//(rank 0 being the best)
template <class T, class Type_Fitness = T>
class indexComparisonStd : public indexComparisonInterface<T, Type_Fitness>
{
public:
	bool operator() (size_t lhs, size_t rhs) { return (this->to_sort->at(lhs).fitness) > (this->to_sort->at(rhs).fitness); }
};


/*#######################################Genetic Optimizer##########################################*/


template
<
	class T,
	class GeneticFunctionsContainerType,
	class Type_Fitness = T
>
class GeneticOptimizer
{
protected:
	GeneticFunctionsContainerType &function_container;
	const GeneticParametersContainer<T, Type_Fitness>  &params_container;
	GeneticVariablesContainer<T, Type_Fitness> &variables_container;


public:
	GeneticOptimizer(GeneticFunctionsContainerType &function_container_, const GeneticParametersContainer<T, Type_Fitness>  &params_container_, GeneticVariablesContainer<T, Type_Fitness>&variables_container_)
		: function_container(function_container_), params_container(params_container_), variables_container(variables_container_)
	{}

	inline bool getOuputsOfIndividualsFrom(Vector<T> &input)
	{
		if (variables_container.tester.getActivation_function() == nullptr || variables_container.tester.getActivation_function() == NULL)
		{
			return false;//this means the  activation function has not been set either through GVC.setsetActivation_function() or GVC.tester.setsetActivation_function()
		}
		IndividualsVector<T, Type_Fitness> &individuals_container = variables_container.individuals_container;
		for (size_t i = 0; i < variables_container.individuals_container.size(); i++)
		{
			variables_container.tester.setLayers(&individuals_container[i].individual);//it wants a pointer
			individuals_container[i].output = variables_container.tester.getResults(input);
		}
		return true;
	}

	void getFitnessOfIndividualsFromOutputs(SimpleFitnessFunctionInterface<T, Type_Fitness> &fitness_functor)
	{
		for (size_t i = 0; i < variables_container.individuals_container.size(); i++)
			variables_container.individuals_container[i].fitness = fitness_functor(variables_container.individuals_container[i].output);
	}

	void getFitnessOfIndividualsFromOutputs(OptimizedFitnessFunctionInterface<T, Type_Fitness> &fitness_functor)
	{
		for (variables_container.index_out = 0; variables_container.index_out < variables_container.individuals_container.size(); variables_container.index_out++)
		{
			fitness_functor(variables_container, params_container);//this one will modify everything directly
		}
	}

	inline void sortFitnessToGenerateRanks()//first action in a cycle: rank every individual according to their fitness
	{
		IndividualsVector<T, Type_Fitness> &individual_containers = variables_container.individuals_container;

		function_container.sorting_index_comparison_function_type.setParameters(variables_container, params_container);

		Vector<size_t> index_to_sort = Vector<size_t>(individual_containers.size(), 0);//make all indices in [0,individual_containers.size()[, that's what we'll sort
		for (size_t i = 0; i < individual_containers.size(); i++)
		{
			index_to_sort[i] = i;//fill the indice
		}
		std::sort(index_to_sort.begin(), index_to_sort.end(), function_container.sorting_index_comparison_function_type);//sort it with a special sorting which translates the index into the fitness and compares afterwards

		for (size_t i = 0; i < individual_containers.size(); i++)
		{
			individual_containers[index_to_sort[i]].rank = i; //set the calculated ranks 
		}
		variables_container.current_max_fit = individual_containers[index_to_sort[0]].fitness;
		variables_container.current_min_fit = individual_containers[index_to_sort[index_to_sort.size() - 1]].fitness;
	}

	//breed_selector =the function used for selecting mates in the mating process
	inline void breed()
	{
		//ATTENTION this is only designed for a to_breed vector of individuals having all the same Network structure( ie number of wieghts and neurons ...)

		IndividualsVector<T, Type_Fitness> &to_breed = variables_container.individuals_container;


		Vector<NetworkStorage<T>> &indiv_copy = variables_container.indiv_copy;//this one has already been initialized by the user, by a call to variables_container.init() or .initCopy()
		for (size_t a = 0; a < to_breed.size(); a++)
		{
			//we have created a (bad for now)iterator for our NetworkStorage just for this moment!
			//for (typename NetworkStorage<T>::iterator i = to_breed[a].individual.begin(); i != to_breed[a].individual.end(); i++)
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
			//hybridation :indiv_copy
			function_container.breeding_mating_function(variables_container, params_container);
		}
	}
	inline void mutate()
	{
		function_container.mutation_function(variables_container, params_container);
	}
};

