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
			//the false tells us that each neuron will not use an additional weight to offset
		NetworkSchema schema(2, sizes, 2,false);	
		// We tell the auto constructor we'll use later that we want to create a net according to our previous schema, and that all weights will be between -1 and 1
		NetworkAutoConstructorParameters<MyTestType> network_structure_(schema, -1, 1);
	
	// II. The activation function (function applied after the summation on each neuron):
		// We'll use a sigmoid function, and we'll make it so that the range is ]-2;2[ 
			// If only one constructor A argument is provided, the range will be ]-A;A[
			// If two (aka A and B) are provided, the formula is B+A / (1 + std::exp( - (in) ) ); therefore the range is ]B;B+A[
			SigmoidFunctorFromTToT<MyTestType> activation_function_sigmoid(2);
		// We just print the range chosen for this function before using it
		std::cout << testFunctionRange<MyTestType, SigmoidFunctorFromTToT<MyTestType>>(activation_function_sigmoid) << std::endl;


	//####################Construction of one neural net####################

	NetworkBuilder<MyTestType> builder;
	// We tell the builder to use as an autoconstructor our network_structure_ and to populate it with random values between those defined earlier in network_structure_
	builder.autoInitializeAndFill(network_structure_);
	// We get back our constructed net
	NetworkStorage<MyTestType> net_1 = builder.construct();	



	//####################Construction of a tester (what gives us the ouput of one net based on the inputs and the weights in the net) for any neural net working on a given type of data####################
	
	TestNetwork<MyTestType> tester;
	// We set its activation function to the sigmoid we defined earlier, it expects a POINTER, since this one will be able to change
	tester.setActivation_function(&activation_function_sigmoid);
	// Print the status of the activation function and the tester
	std::cout << tester.to_string() << std::endl;



	NetworkAutoConstructorParameters<MyTestType> network_structure(schema,-1,1);
	//
	IndividualContainer<MyTestType> ic;
	GeneticParametersContainer<MyTestType> GPC(network_structure,87);
	GeneticVariablesContainer<MyTestType> GVC;
	GVC.init(100, network_structure);

	size_t sz = GVC.individuals_container.size();

	using GenFunctionType =
		GeneticFunctionsContainer
		<
		MyTestType,
		indexComparisonStd<MyTestType>,
		GeneticBreedFitnessBasedSelector<MyTestType>,
		GeneticBreedUniformMating<MyTestType>,
		GeneticMutationDoubleRank<MyTestType>
		> ;

	GenFunctionType GFC;

	GeneticOptimizer<MyTestType, GenFunctionType> GO(GFC, GPC, GVC);
	
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


