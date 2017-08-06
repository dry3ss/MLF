#pragma once

#include "GeneticParametersContainer.h"
#include "GeneticVariablesContainer.h"

/*#######################################Breeding Selector##########################################*/
//This is the function that will be used to
//determine which individual will be coupled each and every time during the mating/breeding
//process. It's job is to set the "partners" attribute of the GeneticVariablesContainer
//with a std::pair containing the indexes of two different individuals to be mated.
template <class T, class Type_Fitness = T>
class GeneticBreedSelectorInterface
{
public:
	virtual inline void operator()(GeneticVariablesContainer<T, Type_Fitness>&variables_container, const GeneticParametersContainer<T, Type_Fitness>  &params_container) = 0;
};
//GeneticBreedFitnessBasedSelector is a simple "rouletter based" selector meaning that
//the chance of being selected for reproduction increases with the fitness(assumed to be
//higher is better) of the individuals.
/* ATTENTION: this class will not work (at least it will be as if the negative values are 0) if you use negative fitness as well as positive !
Also note: Fitness is assumed to be higher=better //TODO make that behaviour configurable
*/
template <class T, class Type_Fitness = T>
class GeneticBreedFitnessBasedSelector : public GeneticBreedSelectorInterface<T, Type_Fitness>
{
protected:
	Vector<Type_Fitness> roulettes_vec;
protected:

	inline size_t getIndexSelectedIndividual(Type_Fitness &rand_)
	{
		Type_Fitness curr_sum = 0;
		for (size_t i = 0; i < roulettes_vec.size(); i++)
		{
			curr_sum += roulettes_vec[i];
			if (rand_ <= curr_sum)
				return i;
		}
		return roulettes_vec.size();
	}
	inline size_t getIndexSelectedIndividual(Type_Fitness &rand_, size_t to_skip)
	{
		Type_Fitness curr_sum = 0;
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
	GeneticBreedFitnessBasedSelector() :roulettes_vec() {}
	virtual inline void operator()(GeneticVariablesContainer<T, Type_Fitness>&variables_container, const GeneticParametersContainer<T, Type_Fitness>  &params_container)
	{
		IndividualsVector<T, Type_Fitness> &to_breed = variables_container.individuals_container;
		//let's define a vector containing the "border values" of our roulette
		roulettes_vec = Vector<Type_Fitness>(to_breed.size(), 0);
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
		size_t indiv1 = getIndexSelectedIndividual(rand_);
		if (indiv1 >= roulettes_vec.size())
			throw std::out_of_range("Did not find our individual 1 in GeneticBreedFitnessBasedSelector::select_two_partners()");
		//once again for the second one, but we have to remove our first one beforehand !
		//TODO debug only:
		Type_Fitness sum_ = sum - roulettes_vec[indiv1];
		uniform_distrib = RandomManagement::RandomEngineGlobal.getUniformDistribution(static_cast<Type_Fitness>(0), sum_);
		rand_ = uniform_distrib(RandomManagement::RandomEngineGlobal.getMT());
		size_t indiv2 = getIndexSelectedIndividual(rand_, indiv1);
		if (indiv2 >= roulettes_vec.size())
			throw std::out_of_range("Did not find our individual 2 in GeneticBreedFitnessBasedSelector::select_two_partners()");
		variables_container.partners = std::pair<size_t, size_t>(indiv1, indiv2);
	}
};
/*#######################################Breeding: Mating##########################################*/
//This is the function that will be applied to determine how
//the selected couple will be "mixed" during the breeding process to obtain the new individual.

template <class T, class Type_Fitness = T>
class GeneticBreedMatingInterface
{
public:
	virtual inline void operator()(GeneticVariablesContainer<T, Type_Fitness>&variables_container, const GeneticParametersContainer<T, Type_Fitness>  &params_container) = 0;
};
//GeneticBreedUniformMating is a simple mating process in which for each weight of the new individual
//one of the two possible weights of its parents is chosen (without change). The proportion of parent1
//over parent2 is determined by the "mating_proportion1_over2" attribute of GeneticParametersContainer
//(default is 0.5, remember that with GeneticBreedFitnessBasedSelector, the first parent has a greater
//chance of having a higher fitness score)
template <class T, class Type_Fitness = T>
class GeneticBreedUniformMating : public GeneticBreedMatingInterface<T, Type_Fitness>
{
public:
	/*ATTENTION this function will only work if all the sizes are correctly configured to be the same*/
	virtual inline void operator()(GeneticVariablesContainer<T, Type_Fitness>&variables_container, const GeneticParametersContainer<T, Type_Fitness>  &params_container)
	{

		const size_t &index_out = variables_container.index_out;
		const std::pair<size_t, size_t> &partners = variables_container.partners;
		Vector<NetworkStorage<T>> &indiv_copy = variables_container.indiv_copy;
		const float &proportion1_over2 = params_container.mating_proportion1_over2;

		IndividualsVector<T, Type_Fitness> &out_container = variables_container.individuals_container;


		typename NetworkStorage<T>::iterator out_iter = out_container[index_out].individual.begin();
		typename NetworkStorage<T>::iterator partner1_iter = indiv_copy[partners.first].begin();
		typename NetworkStorage<T>::iterator partner2_iter = indiv_copy[partners.second].begin();
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