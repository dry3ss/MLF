#pragma once


#include "GeneticParametersContainer.h"
#include "GeneticVariablesContainer.h"

/*#######################################Mutation##########################################*/
//This is the function that will be applied to each and every
//individual to mutate its weights, it provides both the selection of which weights (optionnaly 
//of individuals) to mutate and how to do this.
template <class T, class Type_Fitness = T>
class GeneticMutationInterface
{
public:
	virtual inline void operator()(GeneticVariablesContainer<T, Type_Fitness>&variables_container, const GeneticParametersContainer<T, Type_Fitness>  &params_container) = 0;
};
//GeneticMutationConst uses a simple constant mutation rate on each and every weight that it finds 
//in the GeneticParametersContainer
template <class T, class Type_Fitness = T>
class GeneticMutationConst : GeneticMutationInterface<T, Type_Fitness>
{
public:
	virtual inline void operator()(GeneticVariablesContainer<T, Type_Fitness>&variables_container, const GeneticParametersContainer<T, Type_Fitness>  &params_container)
	{
		const float mut_rate = params_container.const_mutation_rate;
		size_t sz = variables_container.individuals_container.size();
		//btw 0 and 1 to choose if we are indeed going to mutate
		auto distrib_selection = RandomManagement::RandomEngineGlobal.getUniformDistribution(0.0, 1.0);
		//this one has the same range we used to construct our neuron's weights
		auto distrib_weights = RandomManagement::RandomEngineGlobal.getUniformDistribution(params_container.schema.min_random_weights, params_container.schema.max_random_weights);
		for (size_t a = 0; a < sz; a++)
		{
			NetworkStorage<T> current_indiv = variables_container.individuals_container[a].individual;
			for (typename NetworkStorage<T>::iterator i = current_indiv.begin(); i != current_indiv.end(); i++)
			{
				double choice = distrib_selection(RandomManagement::RandomEngineGlobal.getMT());
				if (choice <= mut_rate)
				{
					*i = distrib_weights(RandomManagement::RandomEngineGlobal.getMT());
				}
			}
		}
	}
};

//here the rate of mutation depends on the rank of the previous individual that was in your spot
//in the last generation, the more well ranked he was, the less mutations you should have
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
		auto distrib_weights = RandomManagement::RandomEngineGlobal.getUniformDistribution(params_container.schema.min_random_weights, params_container.schema.max_random_weights);
		for (size_t a = 0; a < sz; a++)
		{
			NetworkStorage<T> &current_indiv = variables_container.individuals_container[a].individual;
			unsigned int current_rank = variables_container.individuals_container[a].rank;
			if (current_rank == 0)
				continue;
			float current_mut_rate = current_rank / static_cast<float>(sz)*max_mut_rate;
			for (typename NetworkStorage<T>::iterator i = current_indiv.begin(); i != current_indiv.end(); i++)
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

//here the values of mutation as well as the rate, both depend on the rank of the previous 
//individual that was in your spot(so not necessarilly any of your parents !) in the last
//generation, the more well ranked he was,the less mutations you should have 
//and the less profound they'll be
template <class T, class Type_Fitness = T>
class GeneticMutationDoubleRank : GeneticMutationInterface<T, Type_Fitness>
{
public:
	virtual inline void operator()(GeneticVariablesContainer<T, Type_Fitness>&variables_container, const GeneticParametersContainer<T, Type_Fitness>  &params_container)
	{
		const float max_mut_rate = params_container.const_mutation_rate;
		size_t sz = variables_container.individuals_container.size();
		auto distrib_selection = RandomManagement::RandomEngineGlobal.getUniformDistribution(0.0, 1.0);
		Type_Fitness random_range = params_container.schema.max_random_weights - params_container.schema.min_random_weights;
		auto distrib_weights = RandomManagement::RandomEngineGlobal.getUniformDistribution(params_container.schema.min_random_weights, params_container.schema.max_random_weights);
		for (size_t a = 0; a < sz; a++)
		{
			unsigned int current_rank = variables_container.individuals_container[a].rank;
			if (current_rank == 0)
				continue;
			NetworkStorage<T> &current_indiv = variables_container.individuals_container[a].individual;
			float current_mut = current_rank / static_cast<float>(sz);
			Type_Fitness current_range = static_cast<Type_Fitness>(random_range*current_mut);//no "overflow" from our range this way
			for (typename NetworkStorage<T>::iterator i = current_indiv.begin(); i != current_indiv.end(); i++)
			{
				double choice = distrib_selection(RandomManagement::RandomEngineGlobal.getMT());
				if (choice <= (current_mut*max_mut_rate))
				{
					Type_Fitness n_a = (*i) - current_range, n_b = (*i) + current_range;
					Type_Fitness n_range = n_b - n_a;
					*i = n_range*distrib_weights(RandomManagement::RandomEngineGlobal.getMT()) + n_a;
				}
			}
		}
	}
};