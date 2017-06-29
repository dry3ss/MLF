#pragma once

#include <fstream>
#include "MLF_GeneticHandler.h"

template <class T>
class quickFileIo
{
private:
	std::string export_file, import_file;
public:
	quickFileIo(const std::string export_file_, const std::string import_file_ = "") :export_file(export_file_)
	{
		if (import_file_ == "")
			import_file = export_file;
	}

	bool exportFile(std::vector<NetworkStorage<T>> &in)
	{
		std::ofstream file(export_file);
		if (!file)
			return false;
		for (size_t i = 0; i < in.size(); i++)
		{
			for (typename NetworkStorage<T>::iterator iter = in[i].begin(); iter != in[i].end(); iter++)
				file << std::to_string(*iter)<<" ";
		}
		file.close();
		return true;
	}


	bool importFileIn(std::vector<NetworkStorage<T>> in)
	{
		std::ifstream file(import_file);
		if (!file)
			return false;
		for (size_t i = 0; i < in.size(); i++)
		{
			for (typename NetworkStorage<T>::iterator iter = in[i].begin(); iter != in[i].end(); iter++)
			{
				T buff=0;
				file >> buff;
				*iter=std::stod(buff);
			}
		}
		file.close();
		return true;
	}
	
	template <class Type_Fitness>
	bool importFileIn(IndividualsVector<T, Type_Fitness> &in)
	{
		std::ifstream file(import_file);
		if (!file)
			return false;
		std::string buff = "";
		for (size_t i = 0; i < in.size(); i++)
		{
			for (typename NetworkStorage<T>::iterator iter = in[i].individual.begin(); iter != in[i].individual.end(); iter++)
			{
				file >> buff;

				*iter = std::stod(buff);
			}
		}
		file.close();
		return true;
	}

};
