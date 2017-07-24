#include "MLF_TestNetwork.h"
#include <iostream>
#include <type_traits>

#include "MLF_GeneticHandler.h" //DEBUG oNLY
#include "quickFIleIO.h"

template<class T>
std::string arrtoString(T arr)
{
	std::string res = "";
	for (size_t i = 0; i < arr.size(); i++)
	{
		res += std::to_string(arr[i]) + " ";
	}
	return res;
}

template <class T, class FunctionType>
std::string testFunctionRange()
{
	FunctionType to_test;
	
	return testFunctionRange<T, FunctionType>(to_test);
}

template <class T, class FunctionType>
std::string testFunctionRange(FunctionType to_test)
{
	std::vector<T> test_function = { -100, -50, -10, -2, -1, 0, 1, 2, 10, 50, 100 };
	std::vector<T>test_function_res(test_function.size(), 0);
	for (size_t i = 0; i < test_function.size(); i++)
		test_function_res[i] = to_test(test_function[i]);
	std::string res = std::string("###Function range test for function type: ") + typeid(FunctionType).name() + "\n";
	for (size_t i = 0; i < test_function.size(); i++)
	{
		res += std::to_string(test_function[i]) + "->" + std::to_string(test_function_res[i]) + "\n";
	}
	return res;
}

//This is a very very simple function that will grow the fitness with the output values.
//Meaning that the network will react by increasing the weights of pretty much each and
//every node.
template<class T>
T fitnessFunction(const std::vector<T> &out)
{
	T r = 0;
	for (size_t i = 0; i < out.size(); i++)
	{
		r += out[i];
	}
	return r;
}

//#define toexport
int main()
{
	//####################Preparation of the neural net(s)####################

	// I. The net's structure:
		// Useful everywhere, the type the net will use, meaning the type each weight, input and output will use: 
		using MyTestType = double;
		// Array containing the number of neurons each layer will have
		std::vector<unsigned int> sizes = { 5,7,6 };	
		// We create a schematic of our neural net : it takes 2 inputs, uses the number of neurons per layer defined beforehand, and gives 2 outputs
		//the false tells us that each neuron will not use an additional weight as an offset
		//here all the weights will be between -1 and 1
		NetworkSchema<MyTestType> schema(2, sizes, 2,false,-1, 1);	

	// II. The activation function (function applied after the summation on each neuron):
		// We'll use a sigmoid function, and we'll make it so that the range is ]-2;2[ 
			// If only one constructor A argument is provided, the range will be ]-A;A[
			// If two (aka A and B) are provided, the formula is B+A / (1 + std::exp( - (in) ) ); therefore the range is ]B;B+A[
			SigmoidFunctorFromTToT<MyTestType> activation_function_sigmoid(2);
		// We just print the range chosen for this function before using it
		std::cout << testFunctionRange<MyTestType, SigmoidFunctorFromTToT<MyTestType>>(activation_function_sigmoid) << std::endl;


//####################Construction of a tester####################
	//(what gives us the ouput of one net based on the inputs and the weights in the net) 
	//it can be used with ANY neural net working on a given type of data (ie float, double ...)
	TestNetwork<MyTestType> tester;
	// We set its activation function to the sigmoid we defined earlier, it expects a POINTER, since this one will be able to change
	tester.setActivation_function(&activation_function_sigmoid);
	// Print the status of the activation function and the tester
	std::cout << tester.to_string() << std::endl;


//@@@ OPTION A: "Testing" (ie: getting the output of a NN based on given inputs) one neural net@@@
	//####################Construction of a (random here) neural net####################
	NetworkBuilder<MyTestType> builder;
	// We tell the builder to use as an autoconstructor our schema and to populate it with random values between those defined earlier in network_structure_
	builder.autoInitializeAndFill(schema);
	// We get back our constructed net
	NetworkStorage<MyTestType> net_1 = builder.construct();	
	//####################Testing the output of our neural net for a given input####################
	std::vector<MyTestType> in = { 0.1,2.5};
	std::vector<MyTestType> out =tester.getResults(in);



	//TODO this is debug
	const std::string file_name="export.txt";
	QuickFileIO<MyTestType> QFIO(file_name);
	std::vector<NetworkStorage<MyTestType>> a;
	a.push_back(net_1);
	QFIO.exportFile(schema,a);
	system("DEBUG: Done exporting");





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

	const GeneticParametersContainer<MyTestType> GPC(schema,87);//It is mandatory
	//to give the network structure (of type NetworkAutoConstructorParameters) so that the algo
	//will know among other things the min&max possible weights
	//Optionnally:
	// -the constant mutation rate in PERCERNTS if GeneticMutationConst is used  (default 1%)
	// -the proportion of weights from parent n°1 over parent n°2 passed on to the child if 
	//  GeneticBreedUniformMating is used (default is half, so that the split is equal)
	//can be set.

	GeneticVariablesContainer<MyTestType> GVC;//We have to create it (no arguments)
	const size_t sz = 100;//size of every generation of the population

	//then either :
	//#Option 1
	GVC.init(sz, schema);//Init with (mandatory):
	// -the number of individuals in the population
	// -the the network structure (of type NetworkAutoConstructorParameters)
	//this will generate a completely new population
	//OR

	// //#Option2 
	// //Import a population from a file then swap the content with our GVC:
	// const std::string file_name="import.txt";
	// NetworkStorage<MyTestType> to_swap;
	// QuickFileIO<MyTestType> QFIO(file_name);
	// QFIO.createAndImportInto(to_swap,schema,sz);
	// GVC.swapIndividualsFrom(to_swap);
	// GVC.initAfterCopy();


	GeneticOptimizer<MyTestType, GenFunctionType> GO(GFC, GPC, GVC);//Now we just create the actual Optimizer

	size_t number_gen = 100;


	Vector<MyTestType> input = { 0.1, 1.1 };

	MyTestType max_fit = 0,prev_max=0,min_fit=0;
	for (size_t gen = 0; gen < number_gen;gen++)
	{ 
		for (size_t i = 0; i < sz; i++)
		{
			tester.setLayers(&GVC.individuals_container[i].individual);
			Vector<MyTestType> outV = tester.getResults(input);
			// III. The fitness function (higher is better) applied to determine which members of the population will successfully reproduce themselves
			GVC.individuals_container[i].fitness = fitnessFunction(outV);
		}


		//TODO debug only:
		prev_max = max_fit;
		GO.generateRanks();
		max_fit = GVC.current_max_fit;
		min_fit = GVC.current_min_fit;
		if (max_fit < prev_max)
			std::cout << "ERREUR max_fit < prev_max" << std::endl;		

		if ((gen+1) % (number_gen / 10) == 0 || (gen==0))
		{
			std::cout << std::to_string((gen + 1) / (number_gen / 100)) + "% " + std::to_string(gen + 1) + "/" + std::to_string(number_gen) + ":" + std::to_string(min_fit)+"->"+ std::to_string(max_fit) << std::endl;
		}
		GO.breed();
		GO.mutate();
	}
	system("pause");
	return 0;
}


