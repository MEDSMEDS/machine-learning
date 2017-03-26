package support_vector_machine_SVM;

import java.util.List;
import java.util.Random;

import javax.print.DocFlavor.BYTE_ARRAY;



public class trainSVM {
	
	
	private Kernel kernel;
	private int N;	
	private double []alph;
	private  int []target;
	private double b=0;//intercept
	
	private double[][] kernelMatrix;//cache the kernel matrix,accelerate the training process
	private double []error_cache;//the second heuristic
	
	
	private double C=0.2;	
	private double tol=0.001;//tolerance
	private double eps=0.001;
	private Random r=new Random();//second heuristic, not bias toward the training data at the beginning 
	
	public int takeStep(int i1,int i2,int y1,double alph1,double E1){//all need shell		
	    if(i1==i2)	        
	        return 0;
	        		
	    double alph2=alph[i2];
	    int y2=target[i2];
	    double E2;
	    if(alph2>0 && alph2<C)
	        E2=error_cache[i2];
	    else
	        E2=learningFun(i2)-y2;	    
	    //%look up alph1,y1,E1,alph2,y2,E2
	    int s=y1*y2;
	    
	    /*String result=compute_LH(alph1,alph2,y1,y2);//%y1,y2,C,a1,a2
	    String []item=result.split(":");
	    double L,H=0;
	    L=Double.parseDouble(item[0]);
	    H=Double.parseDouble(item[1]);	   
	    */
	    double gamma=0;
		double L,H=0;
	   if(y1==y2){
	        gamma=alph1+alph2;
	        if(gamma>C){
	            L=gamma-C;
	            H=C;
	        }
	        else{
	            L=0;
	            H=gamma;
	       }
	   }
	   else{
	        gamma=alph1-alph2;  //%bug not detected
	        if(gamma>0){
	            L=0;
	            H=C-gamma;
	        }
	        else{
	            L=-gamma;
	            H=C;
	        }
	   }
	  
	    if(L==H)	//does not satisfy the constraint that a1+a2=r        
	        return 0;	    
	  
	    double k11=kernelMatrix[i1][i1];
	    double k12=kernelMatrix[i1][i2];
	    double k22=kernelMatrix[i2][i2];
	    double eta=2*k12-k11-k22;
	 
	    double a1=0,a2=0;
	    if(eta<0){
	
	        a2=alph2+y2*(E2-E1)/eta;
	        if(a2<L)
	            a2=L;
	        else if(a2>H)
	            a2=H;
	        
	    }
	    else{
	      
	        double c1=eta/2;
	        double c2=y2*(E1-E2)-eta*alph2;
	        double Lobj=c1*L*L+c2*L;
	        double Hobj=c1*H*H+c2*H;
	        if(Lobj>Hobj+eps)
	            a2=L;
	        else if(Lobj<Hobj-eps)
	            a2=H;
	        else
	            a2=alph2;
	       
	    }
	    
	    if(a2<1e-7)
	        a2=0;
	    else if(a2>C-(1e-7))
	        a2=C;	    
     
	    if(Math.abs(a2-alph2)<eps*(a2+alph2+eps))//%shao l alph h	?????	        
	        return 0;
	    
	    a1=alph1-s*(a2-alph2);//
	    
	    double t;
	    if(a1<0){
	        a2=a2+s*a1;
	        a1=0;
	    }
	    else if(a1>C){
	        t=a1-C;
	        a2=a2+s*t;
	        a1=C;
	    }
	    
	    double bnew=0;
	    if(a1>0 && a1<C)
	        bnew=b-E1-y1*(a1-alph1)*k11-y2*(a2-alph2)*k12;
	    else if(a2>0 && a2<C)
	        bnew=b-E2-y1*(a1-alph1)*k12-y2*(a2-alph2)*k22;
	    else{
	        double b1=b-E1-y1*(a1-alph1)*k11-y2*(a2-alph2)*k12;
	        double b2=b-E2-y1*(a1-alph1)*k12-y2*(a2-alph2)*k22;
	        bnew=(b1+b2)/2;
	    }
	    double delta_b=bnew-b;	    
	    b=bnew;
	    //thresholdU();
	    //error_cacheU(db);
	    double t1=y1*(a1-alph1);
	    double t2=y2*(a2-alph2);
	   // if(bugi<20)
	    	//System.out.println("t1:"+t1+" ,t2:"+t2);
	   // System.out.println("delta_b:"+delta_b+",t1:"+t1+",t2:"+t2);
	    for(int i=0;i<N;i++){
	        if(0<alph[i] && alph[i]<C)
	            error_cache[i]=error_cache[i]+t1*kernelMatrix[i1][i]+t2*kernelMatrix[i2][i]+delta_b;//%puls,not minus
	    }
	    error_cache[i1]=0;
	    error_cache[i2]=0;
	    //error_cache
	    alph[i1]=a1;
	    alph[i2]=a2;
	// %   fprinf('hi');
	    return 1;               //%serious bug, forget three step
	}
	
	public int examineExample(int i1){
		int y1=target[i1];
	    double alph1=alph[i1];//%alph
	    double E1=0;
	    if(alph1>0 && alph1<C)
	        E1=error_cache[i1];
	    else
	        E1=learningFun(i1)-y1;//%learned_func. compromise int i1,int i2,int y1,double alph1,double E1	    
	    double r1=y1*E1;//the kkt condition    
	    double tmax=0,temp=0;int i2=-1,k0=0;
	    if((r1<-tol && alph1<C) || (r1>tol && alph1>0)){// if alpha1 violates the kkt condition, then we can try to optimize a1 and a a2
	    	//planA        	     
	    	//choose alpha2 that maximize the size of the step, 
	    	//approximates the step size by the absolute value of the numerator in equation |E1-E2|    	
	        for(int k1=0;k1<N;k1++){
	            if(alph[k1]>0 && alph[k1]<C)
	                temp=Math.abs(error_cache[k1]-E1);
	                if(temp>tmax){
	                    tmax=temp;
	                    i2=k1;
	                }	            
	        }
	        if(i2>=0)
	            if(takeStep(i1,i2,y1,alph1,E1)==1){
	                return 1;
	            }
	        //planA
	        
	        //planB
	        k0=r.nextInt(N);// not bias toward the beginning of data
	        
	        for(int k1=k0;k1<N+k0;k1++){
	            i2=k1%N;	
	            
	            if(alph[i2]>0 && alph[i2]<C && takeStep(i1,i2,y1,alph1,E1)==1)	                
	            	return 1;	                		               
	        }
	        // planB
	        
	        // planC
	        k0=r.nextInt(N);
	        
	        for(int k1=k0;k1<N+k0;k1++){
	        	i2=k1%N;
	          
	            if(takeStep(i1,i2,y1,alph1,E1)==1)	            	
	                return 1;    
	            
	        }
	        //planC
	    }
	    return 0;
	}
	
	
	public void run(){//first alpha heuristic
		int numChanged=0;
		boolean examineAll=true;
		int count=0;
		while(numChanged>0||examineAll){
		    numChanged=0;
		    if(examineAll){	        //heuristic one
		        for(int k=0;k<N;k++)
		            numChanged=numChanged+examineExample(k);
		    }
		    else
		        for(int k=0;k<N;k++){
		            if(alph[k]!=0 && alph[k]!=C)//alph(1,k) bug ;alph(k)~=0 bug                 
		                numChanged=numChanged+examineExample(k);                 	 
		        }
		   //System.out.println("num:"+numChanged+" , examineAll:"+examineAll);
		    //System.out.println("count:"+count);
		    if(examineAll)//only once, if not changed,break		        
		        examineAll=false;		    
		    else if(numChanged==0)//plan b not changed, then plan a
		        examineAll=true;		      		    
		    else;	//still plan B      
		    //count++;
		    
		}		
		//System.out.println("count"+count++);
	}
	
	
	
	
	public double learningFun(int k){ //learning function
		double res=0;		
		for(int i=0;i<N;i++)			
			res+=alph[i]*target[i]*kernelMatrix[i][k];				
		return res+b;
	}
	
	private  void getKernelMatrix(List<Store>[] training_samples){ //get all dot_product
		//long time=System.currentTimeMillis();
		//System.out.println("Calculating kernel matrix...");		
		kernelMatrix=new double[N][N];
		for(int i=0;i<N;i++)
			kernelMatrix[i]=new double[N];
		for(int i=0;i<N;i++){				
			for(int j=i;j<N;j++){				
				kernelMatrix[i][j]=kernel.kernelFun(training_samples[i],training_samples[j]);
				kernelMatrix[j][i]=kernelMatrix[i][j];
			}
		}
		//System.out.println("Calculating kernel matrix...finished!!!"+ "  "+(System.currentTimeMillis()-time)/1000);
	}
	
	public void test(){
		double err=0;
		for(int i=0;i<N;i++){
			if(learningFun(i)*target[i]<0)
				err++;
		}
		System.out.println("training data error rate:"+err/N);
	}
	
	public  void train(int len,double[] alpha,int []targt,List<Store>[] training_samples,double []b,double C,double tol,double eps,int index,Kernel kernel){
		this.N=len;
		this.alph=alpha;
		this.target=targt;
		this.C=C;
		this.tol=tol;
		this.eps=eps;
		this.kernel=kernel;
		error_cache=new double[N];
		getKernelMatrix(training_samples);
		run();
		b[index]=this.b;
		test();
	}
}
