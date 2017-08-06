#pragma once
#include "MLF_NetworkStorage.h"


//This second one contains constant parameters, the container itself is passed as a reference to constant, 
//not necessarilly useful for most functions, but if the user wants to inherit this class and add his own, he can !

template
<
	class T, class Type_Fitness = T
>
struct GeneticParametersContainer
{
	NetworkSchema<T> schema;
	//for mating:
	const float mating_proportion1_over2;
	//for mutation
	float const_mutation_rate;

	GeneticParametersContainer(const NetworkSchema<T> &schema_, const float prop_1_over_2_in_percents = 50, const float const_mutation_rate_percents = 1)
		:schema(schema_), mating_proportion1_over2(prop_1_over_2_in_percents / 100.0), const_mutation_rate(const_mutation_rate_percents / 100.0)
	{
		//const_mutation_rate = const_mutation_rate_percents*schema.getTotalNumberNeurons() / 100.0;
	}
};