#pragma once

#include <fstream>
#include <string>
#include "GeneticVariablesContainer.h"


template <class T>
class QuickFileIO
{
private:
	std::string export_file, import_file;
public:
	QuickFileIO(const std::string export_file_, const std::string import_file_ = "") :export_file(export_file_),import_file(import_file_)
	{
		if (import_file_ == "")
			import_file = export_file;
	}

	bool exportFile(const NetworkSchema<T> &schema, std::vector<NetworkStorage<T>> &in)
	{
		std::ofstream file(export_file);
		file.precision(17);
		if (!file)
			return false;
		file << "Schema of individuals: " << schema.toStringForExport() << std::endl;
		file << "Number of individuals: " << in.size() << std::endl;
		for (size_t i = 0; i < in.size(); i++)
		{
			for (typename NetworkStorage<T>::iterator iter = in[i].begin(); iter != in[i].end(); iter++)
				file << *iter << " ";
			file <<"\n";
		}
		file.close();
		return true;
	}
	template <class Y>//Y has no importance here
	bool exportFile(const NetworkSchema<T> &schema, IndividualsVector<T, Y> &in)
	{
		std::ofstream file(export_file);
		file.precision(20);
		if (!file)
			return false;
		file << "Schema of individuals: " << schema.toStringForExport() << std::endl;
		file << "Number of individuals: " << in.size() << std::endl;
		for (size_t i = 0; i < in.size(); i++)
		{
			for (typename NetworkStorage<T>::iterator iter = in[i].individual.begin(); iter != in[i].individual.end(); iter++)
				file <<*iter << " ";
			file << "\n";
		}
		file.close();
		return true;
	}

	bool retrieveInitializeAndImportInto(std::vector<NetworkStorage<T>> &in,NetworkSchema<T> &schema, size_t &size_pop)
	{
		bool a=retrieveSchemaAndSizePopFromFile(schema, size_pop);
		if (!a)
			return a;
		NetworkBuilder<T>::initializeZeroesWholePop(in, schema, size_pop);
		return importPopFromFileIn(in);
	}
	
static std::vector<std::string> splitOn(const std::string &in, const std::string &delims)
{
	std::vector<std::string> r;
	const size_t sz_delims = delims.size();
	const size_t sz_in = in.size();
	size_t start_from = 0;
	for (size_t i = 0; i < sz_in; i++)
	{
		for (size_t j = 0; j < sz_delims; j++)
		{		
			if (in[i] == delims[j])
			{
				if(i!= start_from)
					r.push_back(in.substr(start_from, i- start_from));
				start_from = i + 1;
				break;
			}
		}
	}
	if (start_from < sz_in)
		r.push_back(in.substr(start_from, sz_in - start_from));
	return r;
}

inline bool retrieveSchemaAndSizePopFromFile(NetworkSchema<T> &schema, size_t &size_pop)
{
	std::ifstream file(import_file);
	file.precision(20);
	if (!file)
		return false;
	std::string buff = "";
	size_t pos = 0;
	//######################################
	//		let's extract the schema
	//######################################
	//format is : 
	//Schema of individuals: [schema.min_random_weights;schema.max_random_weights]:schema.nb_input; ...schema.neurons_per_layer[]..   ;schema.nb_output
	//example : 
	//Schema of individuals: [-1.000000;1.000000]:2;5,7,6;2
	getline(file, buff);
	pos = buff.find(": ");
	if (pos == std::string::npos || pos >= buff.size())
		return false;
	buff = buff.substr(pos + 2);
	std::vector<std::string> splitted = splitOn(buff, "[];:");
	if (splitted.size() < 4)//there is not even min, max, nb_in, nb_out
		return false;

	size_t i = 0;
	//min_random_weights
	schema.min_random_weights = static_cast<T>(stod(splitted[i]));
	i++;
	//max_random_weights
	schema.max_random_weights = static_cast<T>(stod(splitted[i]));
	i++;
	//nb_input
	schema.nb_input = stod(splitted[i]);
	i++;
	//nb_output
	schema.nb_output = stod(splitted[splitted.size() - 2]);// NO i++ this time
	//nb_output
	schema.affine = (splitted[splitted.size() - 1])=="Y";// NO i++ this time
	//neurons_per_layer
	splitted = splitOn(splitted[i], ",");
	std::vector<unsigned int> buff_sizes;
	for (i = 0; i < splitted.size(); i++)
		buff_sizes.push_back(stoi(splitted[i]));
	schema.neurons_per_layer = buff_sizes;

	std::cout << "Imported schema:\n" << schema.toStringVerbose();
	//######################################
	//		The number of individuals
	//######################################
	std::getline(file, buff);
	pos = buff.find(": ");
	if (pos == std::string::npos || pos >= buff.size())
		return false;
	buff = buff.substr(pos + 2);
	size_pop = stoi(buff);
	std::cout << "Imported size of the population:" << size_pop << "\n";
	return true;
}


inline	bool importPopFromFileIn(std::vector<NetworkStorage<T>> &in)
	{
		std::ifstream file(import_file);
		file.precision(20);
		if (!file)
			return false;
		std::string useless="";
		std::getline(file, useless);
		std::getline(file, useless);//twice to skip the first 2 lines
		//######################################
		//		The content
		//######################################
		for (size_t i = 0; i < in.size(); i++)
		{
			for (typename NetworkStorage<T>::iterator iter = in[i].begin(); iter != in[i].end(); iter++)
			{
				T buff=0;
				file >> buff;
				*iter=buff;
			}
		}
		file.close();
		return true;
	}
};
