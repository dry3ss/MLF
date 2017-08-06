#pragma once


//This first one contains all the functions that the user can choose to change, for each of them, an operator() must exist! 
//Also the whole point of having  GeneticParametersContainer afterwards is that none of their constructors should take any parameters, or at least a defautl one should exist
template
<
	class T,
	class SortingIndexComparisonFunctionType,
	class BreedingSelectionFunctionType,
	class BreedingMatingFunctionType,
	class MutationFunctionType,
	class Type_Fitness = T
>
struct GeneticFunctionsContainer
{
	//static_assert(std::is_same<     decltype(BreedingSelectionFunctionType::func()), void>::value, "BreedingSelectionFunctionType must have a \"func()\" function");
	SortingIndexComparisonFunctionType sorting_index_comparison_function_type;//ATTENTIOn this one works differently, its constructor is called with the GeneticContainers, and the operator() looks like this: operator() (size_t lhs, size_t  rhs) 
	BreedingSelectionFunctionType breeding_selection_function;
	BreedingMatingFunctionType breeding_mating_function;
	MutationFunctionType mutation_function;
	GeneticFunctionsContainer() :
		sorting_index_comparison_function_type(), breeding_selection_function(), breeding_mating_function(), mutation_function() {}
	std::string to_string()
	{
		return std::string("breeding_mating_function::") + typeid(breeding_mating_function).name()
			+ "\nbreeding_selection_function::" + typeid(breeding_selection_function).name()
			+ "\nsorting_index_comparison_function_type::" + typeid(sorting_index_comparison_function_type).name()
			+ "\nmutation_function::" + typeid(mutation_function).name();
	}
};