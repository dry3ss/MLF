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

template <class T,class FunctionType>
std::string testFunctionRange()
{
	FunctionType to_test;
	std::vector<T> test_function = { -100, -50, -10, -2, -1, 0, 1, 2, 10, 50, 100 };
	std::vector<T>test_function_res(test_function.size(), 0);
	for (size_t i = 0; i < test_function.size(); i++)
		test_function_res[i] = to_test(test_function[i]);
	std::string res = std::string("###Function range test for function type: ") + typeid(FunctionType).name()+"\n";
	for (size_t i = 0; i < test_function.size(); i++)
	{
		res += std::to_string(test_function[i]) + "->" + std::to_string(test_function_res[i]) + "\n";
	}
	return res;
}


template<class T>
T costFunction(const std::vector<T> &out)
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
	using MyTestType = double;

	std::vector<unsigned int> sizes = { 5,7,6 };


	NetworkSchema schema(2, sizes, 2,false);
	NetworkSchema schema_aff(2, sizes, 2,true);

	NetworkAutoConstructorParameters<MyTestType> network_structure_(schema, -1, 1);
	NetworkAutoConstructorParameters<MyTestType> network_structure_aff(schema_aff, -1, 1);

	NetworkBuilder<MyTestType> builder;
	builder.autoInitializeAndFill(network_structure_);
	NetworkStorage<MyTestType> net_1 = builder.construct();
	builder.autoInitializeAndFill(network_structure_aff);
	NetworkStorage<MyTestType> net_aff = builder.construct();
	net_1;

	SigmoidFunctorFromTToT<MyTestType> sig(2);

	std::cout << testFunctionRange<MyTestType, SigmoidFunctorFromTToT<MyTestType>>()<<std::endl;


	TestNetwork<MyTestType> tester;
	tester.setActivation_function(&sig);
	std::cout << tester.to_string() << std::endl;


	Vector<MyTestType> in = { 0.1, 1.1 };

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

	MyTestType max_fit = 0,prev_max=0,min_fit=0;
	for (size_t gen = 0; gen < number_gen;gen++)
	{ 
		for (size_t i = 0; i < sz; i++)
		{
			tester.setLayers(&GVC.individuals_container[i].individual);
			Vector<MyTestType> outV = tester.getResults(in);
			GVC.individuals_container[i].fitness = costFunction(outV);
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


