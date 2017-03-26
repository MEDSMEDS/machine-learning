package support_vector_machine_SVM;

import java.util.List;

public class Mathtool {
	
	public static double kernel(double dot_product){//kernel_function
		return Math.pow((dot_product+1), 11);
	}	
	
	public static double dot_product(List<Store> sample1,List<Store> sample2) //basic dot_product
	{
		double sum = 0;
		int xlen = sample1.size();
		int ylen = sample2.size();
		int i = 0;
		int j = 0;
		while(i < xlen && j < ylen)
		{
			if(sample1.get(i).index == sample2.get(j).index)
				sum += sample1.get(i++).value * sample2.get(j++).value;
			else
			{
				if(sample1.get(i).index > sample2.get(j).index)
					++j;
				else
					++i;
			}
		}
		return sum;
	}
}
