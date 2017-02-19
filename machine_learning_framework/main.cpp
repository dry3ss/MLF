#include "MLF_TestNetwork.h"
#include <iostream>

template<class T>
std::string arrtoString(T arr)
{
	std::string res = "";
	for (int i = 0; i < arr.size(); i++)
	{
		res += std::to_string(arr[i]) + " ";
	}
	return res;
}



template <class T ,template<class, class> class V1=std::vector, class A1 = std::allocator<T>, template<class, class> class V2 = V1, class A2 = A1, template<class, class> class V3 = V1, class A3 = A1 >
using myMatrix= V1<V2<V3<T,A3>,A2>,A1>;


int main()
{
	myMatrix<float> lala;
	//myvect<float, std::vector> b;
	//myvect<std::vector,float,std::allocator<float>> a;
	system("pause");
	return 0;
}




/*

int main()
{
	NetworkBuilder<float> builder;
	std::vector<int> sizes = {3,2};
	builder.initialize(2, sizes,1);

	
	builder.access().content[0][0][0] = 1;
	builder.access().content[0][0][1] = 1;
	builder.access().content[1][0][0] = 1;
	builder.access().content[2][0][0] = 1;
	
	NetworkStorage<VectorMatrix3D<float>> network = builder.construct();
	LinearFunctorFromTToT<float> lin;
	TestNetwork<float> lala;
	lala.setLayers(&network);
	lala.setActivation_function(&lin);
	std::cout<<lala.to_string()<<std::endl;


	Vector<float> in = { 1,2 };
	std::cout <<"in:\n"+ arrtoString(in) << std::endl;
	Vector<float> out= lala.getResults(in);
	std::cout << "out:\n" + arrtoString(out) << std::endl;


	system("pause");
	return 0;
}
*/

