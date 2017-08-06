#include "MLF_TestNetwork.h"
#include <iostream>
#include <type_traits>

#include "GeneticOptimizer.h"
#include "quickFIleIO.h"
#include "MLF_Utilities.h"



//This is a very very simple function that will grow the fitness with the output values.
//Meaning that the network will react by increasing the weights of pretty much each and
//every node.

template <class T, class Type_Fitness = T>
class MySimpleFitnessFunction : public SimpleFitnessFunctionInterface<T, Type_Fitness>
{
public:
	virtual inline Type_Fitness operator()(const Vector<T> &out)
	{
		T r = 0;
		for (size_t i = 0; i < out.size(); i++)
		{
			r += std::max(static_cast<T>(0), out[i]);
		}
		if (r != 0)
			r= 1 / r;
		return r;
	}
};

template <class T, class Type_Fitness = T>
class MyOptimizedFitnessFunction : public OptimizedFitnessFunctionInterface<T, Type_Fitness>
{
public:
	virtual inline Type_Fitness operator()(GeneticVariablesContainer<T, Type_Fitness>&variables_container, const GeneticParametersContainer<T, Type_Fitness>  &params_container)
	{
		T &r = variables_container.individuals_container[variables_container.index_out].fitness;
		const Vector<T> &out = variables_container.individuals_container[variables_container.index_out].output;
		r = 0;
		for (size_t i = 0; i < out.size(); i++)
		{
			r += std::max(static_cast<T>(0), out[i]);
		}
		if (r != 0)
			r = 1 / r;	
		return r;
	}
};

int main()
{

	using namespace mlf_utils_namespace;
	std::cout.precision(17);//the max precision for doubles;
	//####################Preparation of the neural net(s)####################

	// I. The net's structure:
		// Useful everywhere, the type the net will use, meaning the type each weight, input and output will use: 
		using MyTestType = double;
		
		const unsigned int size_in = 2, size_out = 2;
		// Array containing the number of neurons each layer will have
		std::vector<unsigned int> sizes = { 2 };//{ 5,7,6 };	
		// We create a schematic of our neural net : it takes 2 inputs, uses the number of neurons per layer defined beforehand, and gives 2 outputs
		//the false tells us that each neuron will not use an additional weight as an offset
		//here all the weights will be between -1 and 1
		NetworkSchema<MyTestType> schema(size_in, sizes, size_out,false,-1, 1);

	// II. The activation function (function applied after the summation on each neuron):
		// We'll use a sigmoid function, and we'll make it so that the range is ]-2;2[ 
			// If only one constructor A argument is provided, the range will be ]-A;A[
			// If two (aka A and B) are provided, the formula is B+A / (1 + std::exp( - (in) ) ); therefore the range is ]B;B+A[
			SigmoidFunctorFromTToT<MyTestType> activation_function_sigmoid(2);
		// We just print the range chosen for this function before using it
		std::cout << testFunctionRange<MyTestType, SigmoidFunctorFromTToT<MyTestType>>(activation_function_sigmoid) << std::endl;

//@@@ OPTION B: Use the GeneticOptimizer to breed&select random neural nets based on chosen parameters@@@
	
//The GeneticOptimizer works as follows:
//- we give it  three containers beforehand:
//		-> The GeneticFunctionsContainer tells it which functions to use when mutating and selecting mates
//			note: for each of them, an operator() must exist! Also the whole point of having a
//				GeneticParametersContainer afterwards is that NONE of their constructors should take 
//				any parameters, or at least a defautl one should exist
//		-> The GeneticParametersContainer tells the previsouly chosen functions which CONSTANT values (those  
//			parameters depend on the functions chosen) to use		
//			note: the container itself is passed as a reference to constant
//			note: depending on the functions chosen (and the way they were coded) it might not necessarilly 
//				be extremely useful but if the user wants to inherit this class and add his own, he can !
//		-> The GeneticVariablesContainer contains everything the GeneticOptimizer and the functions
//			chosen inside the GeneticFunctionsContainer will manipulate and change throughout the algorithm


	using GenFunctionType =
		GeneticFunctionsContainer
		<
		MyTestType,
		indexComparisonStd<MyTestType>,// This is the comparator obj that will be used to rank all individuals 
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
		//the fitness, the "better" it will be ranked, if we want "lower is better" behaviour another comparator class
		//using "<"" instead of ">" is needed

		GeneticBreedFitnessBasedSelector<MyTestType>,//This is the function that will be used to
		//determine which individual will be coupled each and every time during the mating/breeding
		//process. It's job is to set the "partners" attribute of the GeneticVariablesContainer
		//with a std::pair containing the indexes of two different individuals to be mated.
		//GeneticBreedFitnessBasedSelector is a simple "rouletter based" selector meaning that
		//the chance of being selected for reproduction increases with the fitness(assumed to be
		//higher is better) of the individuals.
		//NOTE: this function will NOT WORK properly (at least it will be as if the negative values are 0)
		// if you use NEGATIVE fitness as well as positive !
		
		GeneticBreedUniformMating<MyTestType>,//This is the function that will be applied to determine how
		//the selected couple will be "mixed" during the breeding process to obtain the new individual.
		//GeneticBreedUniformMating is a simple mating process in which for each weight of the new individual
		//one of the two possible weights of its parents is chosen (without change). The proportion of parent1
		//over parent2 is determined by the "mating_proportion1_over2" attribute of GeneticParametersContainer
		//(default is 0.5, remember that with GeneticBreedFitnessBasedSelector, the first parent has a greater
		//chance of having a higher fitness score)

		GeneticMutationDoubleRank<MyTestType>//This is the function that will be applied to each and every
		//individual to mutate its weights, it provides both the selection of which weights (optionnaly 
		//of individuals) to mutate and how to do this.
		//With GeneticMutationDoubleRank the values of mutation as well as the rate, both depend on the rank
		//of the previous individual that was in your spot(so not necessarilly any of your parents !)
		//in the last generation, the more well ranked he was,the less mutations you should have 
		//and the less profound they'll be
		> ;

		//TODO be able to add the "const" keyword without issue...
	 GenFunctionType GFC;//Now that we've decided what it will contain, we can declare it

	const GeneticParametersContainer<MyTestType> GPC(schema,50,80);//It is mandatory
	//to give the network structure (of type NetworkAutoConstructorParameters) so that the algo
	//will know among other things the min&max possible weights
	//Optionnally:	
	// -the proportion of weights from parent n°1 over parent n°2  in PERCERNTS passed on to the child if 
	//  GeneticBreedUniformMating is used (default is half, so that the split is equal
	// -the constant mutation rate in PERCERNTS if GeneticMutationConst is used  (default 1%)
	// note that it is also used in GeneticMutationSimpleRank and GeneticMutationDoubleRank
	// to set the MAXIMUM rate of mutation
	//can be set.

	GeneticVariablesContainer<MyTestType> GVC;//We have to create it (no arguments)

	const size_t sz_1_gen = 100;//size of every generation of the population
	GVC.initAndGenerateNewPop(sz_1_gen, schema);//Init with (mandatory):
	// -the number of individuals in the population
	// -the the network structure (of type NetworkAutoConstructorParameters)
	//this will have generated a completely new population

	//the tester inside GVC is what gives us the ouput of one net based on the inputs and the weights in the net
	//it can be used with ANY NN working on a given type of data (ie float, double ...)
	// We set its activation function to the sigmoid we defined earlier, it expects a POINTER, since this one will be able to change
	GVC.setTesterActivation_function(&activation_function_sigmoid);
	
	GeneticOptimizer<MyTestType, GenFunctionType> GO(GFC, GPC, GVC);//Now we just create the actual Optimizer

	const size_t number_gen = 1000;
	MyTestType &max_fit = GVC.current_max_fit, &min_fit = GVC.current_min_fit;//for printing only

	Vector<MyTestType> input = { 1 ,1 };//this is the arbitrary input that we will use to test all our networks

	//I created two fitness function performing exactly the same job to showcase how the "optimized" version is created
	//any of the two can be used here even though the optimized is well, optimized and should perform slightly better (at least it did during my tests)
	//MySimpleFitnessFunction<MyTestType> fitnessFunction;
	MyOptimizedFitnessFunction<MyTestType> fitnessFunction;	
	bool ok = true;
	for (size_t gen = 0; gen < number_gen;gen++)
	{ 
		ok=GO.getOuputsOfIndividualsFrom(input);//the GO will use the tester setup previsouly in the GVC to get the output
		//of every NN based on the provided input and thus populate GVC.individuals_container[i].ouput
		if (!ok)
			return 0;

		GO.getFitnessOfIndividualsFromOutputs(fitnessFunction);//using this fitness function, the GO gets the fitness of 
		//the previously calculated outputs and puts it in GVC.individuals_container[i].fitness

		GO.sortFitnessToGenerateRanks();//usingt std::sort and a modified comparison operator, the GO "sorts" the indexes
		//of the individuals based on the fitness scores of those individuals and thus fill GVC.individuals_container[i].rank

		std::cout<<toString(gen, number_gen, GVC);//This function is clearly not very good performance wise, but it shouldn't be called to often to cause any real problem
		GO.breed();
		GO.mutate();		
	}

	//let's rank one last time to see who's the best at the very last generation 
	ok=GO.getOuputsOfIndividualsFrom(input);
	if (!ok)
		return 0;
	GO.getFitnessOfIndividualsFromOutputs(fitnessFunction);
	GO.sortFitnessToGenerateRanks();


	system("pause");
	//let's see the output of the very best one
	IndividualContainer<MyTestType> &best = GVC.getIndividualWithRank();//gets us the NN with ranked 0 (ie best fitness)
	GVC.tester.setLayers(&best.individual);//we tell the tester that this is the individual we want to work on
	std::vector<MyTestType> out_ = GVC.tester.getResults(input);
	std::cout << toString(out_);
	system("pause");
	//#############################Export to file###################
	//let's export our results to a file
	std::string file_name = "export.txt";
	QuickFileIO<MyTestType> QFIO (file_name);
	ok=QFIO.exportFile(GPC.schema, GVC.individuals_container);
	std::cout << "Export: " << (ok ? "Successful" : "Unsuccessful") << "\n";
	system("pause");
	return 0;
}


