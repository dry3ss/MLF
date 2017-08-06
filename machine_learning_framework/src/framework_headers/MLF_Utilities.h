#pragma once
#include <string>
#include <sstream>
namespace mlf_utils_namespace
{ 
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

		std::stringstream res;
		res.precision(17);
		res << "###Function range test for function type: " << typeid(FunctionType).name() << "\n";
		for (size_t i = 0; i < test_function.size(); i++)
		{
			res << test_function[i] << "->" << test_function_res[i] << "\n";
		}
		return res.str();
	}

	template <class T>
	std::string toString(const T &in)
	{
		return in.toString();
	}
	template <>
	std::string toString<double>(const double &in)
	{
		std::stringstream res;
		res.precision(17);
		res<<in;
		return res.str();
	}
	template <class T>
	std::string toString(const std::vector<T> &in)
	{
		std::string r = "";
		std::string nb = std::to_string(in.size());
		for (size_t i = 0; i < in.size(); i++)
		{
			r.append(std::to_string(i + 1) + "/" + nb + ":\n\t");
			r.append(toString(in[i]) + "\n");
		}
		return r;
	}
	
	template
		<
			class T, class Type_Fitness = T
		>
	//This function is clearly not very good performance wise, but it shouldn't be called to often to cause any real problem
	std::string toString(const size_t &gen, const size_t &number_gen, const GeneticVariablesContainer<T, Type_Fitness> &GVC)
	{
		const size_t number_gen_div10 = (number_gen / 10 > 1) ? number_gen / 10 : 1;
		const size_t number_gen_div100 = (number_gen / 100 > 1) ? number_gen / 100 : 1;
		const T &max_fit = GVC.current_max_fit, &min_fit = GVC.current_min_fit;
		std::stringstream res;
		res.precision(17);
		if ((gen + 1) % (number_gen_div10) == 0 || (gen == 0))
		{
			res << std::to_string((gen + 1) / (number_gen_div100)) + "% (" + std::to_string(gen + 1) + "/" + std::to_string(number_gen) + ") min:" << min_fit << " -> max:" << max_fit << std::endl;
		}
		return res.str();
	}
}
