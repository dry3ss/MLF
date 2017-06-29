
#include <random>
#include <limits>
#include <cmath>// std::nextafter

//debug:
#include <iostream>
#include <string>

namespace RandomManagement
{
	class RandomEngine
	{
	private:
		std::random_device rd;
		std::mt19937 mt;           
	public:
		RandomEngine() :rd()//,mt(rd()) for debug removed and seeded manually !
		{
			//unsigned int seed=rd();
			unsigned int seed = 57776542;
			std::cout << "SEED:" + std::to_string(seed) << std::endl;
			mt = std::mt19937(seed);
			
		}
		std::random_device& getRD() 
		{ return rd; }
		std::mt19937& getMT() 
		{ return mt; }

		//since uniform_real_distribution doesn't work with integral types and vice versa with uniform_int_distribution
		//let's use SFINAE to get the good one automatically no matter our type

		template <class T>
		typename std::enable_if< std::is_integral<T>::value, std::uniform_int_distribution<T> >::type
			getUniformDistribution(const T &min, const T &max)
		{
			return std::uniform_int_distribution<T>(min, max);
		}

		template <class T>
		typename std::enable_if< std::is_floating_point<T>::value, std::uniform_real_distribution<T> >::type
			getUniformDistribution(const T &min, const T &max)
		{
				return std::uniform_real_distribution<T>(min, max);
		}

	};
	RandomEngine RandomEngineGlobal;//global variable, we'll only need one, but we don't NEED to enforce this condition, so no point making it a singleton
}



