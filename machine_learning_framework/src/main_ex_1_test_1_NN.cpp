#include "MLF_TestNetwork.h"
#include <iostream>
#include <type_traits>

#include "GeneticOptimizer.h"
#include "quickFIleIO.h"
#include "MLF_Utilities.h"



//This is a very very simple function that will grow the fitness with the output values.
//Meaning that the network will react by increasing the weights of pretty much each and
//every node.
template<class T>
T fitnessFunction(const std::vector<T> &out)
{
	T r = 0;
	for (size_t i = 0; i < out.size(); i++)
	{
		r += std::max(static_cast<T>(0), out[i]);
	}
	if (r != 0)
		return 1 / r;
	else
		return r;
}



int main()
{

	using namespace mlf_utils_namespace;

	//####################Preparation of the neural net(s)####################

	// I. The net's structure:
		// Useful everywhere, the type the net will use, meaning the type each weight, input and output will use: 
		using MyTestType = double;
		// Array containing the number of neurons each layer will have
		const unsigned int size_in = 2, size_out = 2;
		std::vector<unsigned int> sizes ={ 5,7,6 };	
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


//####################Construction of a tester####################
	//(what gives us the ouput of one net based on the inputs and the weights in the net) 
	//it can be used with ANY neural net working on a given type of data (ie float, double ...)
	TestNetwork<MyTestType> tester;
	// We set its activation function to the sigmoid we defined earlier, it expects a POINTER, since this one will be able to change
	tester.setActivation_function(&activation_function_sigmoid);
	// Print the status of the activation function and the tester
	//it will tell us that the funciton has been set but no neural net to work on is defined
	std::cout << tester.toString() << std::endl;


//@@@ OPTION A: "Testing" (ie: getting the output of a NN based on given inputs) one neural net@@@
	//####################Construction of a (random here) neural net####################
	NetworkBuilder<MyTestType> builder;
	// We tell the builder to use as an autoconstructor our schema and to populate it with random values between those defined earlier in network_structure_
	builder.autoInitializeAndFill(schema);
	// We get back our constructed net
	NetworkStorage<MyTestType> net_1 = builder.construct();	
	//####################Testing the output of our neural net for a given input####################
	std::vector<MyTestType> in = { 0.1,2.5};
	tester.setLayers(&net_1);
	std::cout << tester.toString() << std::endl;//this time it's ok
	std::vector<MyTestType> out =tester.getResults(in);

	std::cout << toString(out);


	system("pause");
	return 0;
}


