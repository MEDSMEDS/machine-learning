package support_vector_machine_SVM;

import java.util.List;

public class Kernel {

	public static final int LINEAR = 0;
	public static final int POLY = 1;
	public static final int RBF = 2;
	public static final int SIGMOID = 3;
	public static final int PRECOMPUTED = 4;
	private int kernel_type=2;
	public int degree=10;	// for poly
	public double gamma;	// for poly/rbf/sigmoid
	public double coef=1;	// for poly/sigmoid
	public double two_sigma_squared=2;
	public Kernel(int type) {		
		kernel_type=type;
	}
	public double kernelFun(List<Store> sample1,List<Store> sample2){//kernel_function
		double dot_product=Mathtool.dot_product(sample1,sample2);
		switch(kernel_type)
		{
			
			case Kernel.LINEAR:
				return dot_product;
			case Kernel.POLY:
				return Math.pow((coef+dot_product), degree);
			case Kernel.RBF:
				return Math.exp(-(Mathtool.dot_product(sample1,sample1)+Mathtool.dot_product(sample2,sample2)-2*dot_product)/two_sigma_squared);
			case Kernel.SIGMOID:
				return Math.tanh(gamma*dot_product+coef);
		//	case trainSVM.PRECOMPUTED:
		//		return x[i][(int)(x[j][0].value)].value;
			default:
				return dot_product;
		}
	}
}	
