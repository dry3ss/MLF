#pragma once

//forward declaration
template
<
	class T, class Type_Fitness = T
>
struct GeneticVariablesContainer;
template
<
	class T, class Type_Fitness = T
>
struct GeneticParametersContainer;


/*#######################################Fitness function interface##########################################*/
//This is the function that will be applied to each and every
//individual's output to determine 
//its fitness score and thus its ranking

//this one is the simplest kind to create , without caring for the internal workings of the GeneticVariablesContainer
//but since the GeneticOptimizer will handle the translation with the GeneticVariablesContainer, it will be less efficient
template <class T, class Type_Fitness = T>
class SimpleFitnessFunctionInterface
{
public:
	virtual inline Type_Fitness operator()(const Vector<T> &out) = 0;
};
// this one is just a tiny bit harder to create, but it reduces the useless coppying involved with the simplest one
//basically three items in the GeneticVariablesContainer will ever be used:
// variables_container.index_out 
// will tell you for which of the individuals we have to find the fitness
// therefore :  variables_container.individuals_container[ variables_container.index_out ] will contain both :
//  - the output of the NN (equivalent to "const Vector<T> &out" in the SimpleFitnessFunctionInterface) : 
// variables_container.individuals_container[ variables_container.index_out ].output
//  - the new fitness  of the NN ( equivalent to the "Type_Fitness" returned in the SimpleFitnessFunctionInterface)
// variables_container.individuals_container[ variables_container.index_out ].fitness
template <class T, class Type_Fitness = T>
class OptimizedFitnessFunctionInterface
{
public:
	virtual inline Type_Fitness operator()(GeneticVariablesContainer<T, Type_Fitness>&variables_container, const GeneticParametersContainer<T, Type_Fitness>  &params_container) = 0;
};

