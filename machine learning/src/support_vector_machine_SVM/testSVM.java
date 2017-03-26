package support_vector_machine_SVM;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;








public class testSVM {
	private int category=10;
	private int comparator_num;
	private  int []training_samples_Labels=new int[4000];
	private  int []test_samples_Labels=new int[1000];
	private ArrayList<Store>[] trainingSamples=new ArrayList[4000];//change
	private ArrayList<Store>[] testSamples=new ArrayList[1000];
	
	//for one versus one training mode
	private double[] b_arr;
	private double [][]alph_arr;
	
	private int [][]target_arr;
	
	private int[][]baseline=new int[10][10];
	
	private int [][]map=new int[45][2];
	private int[][]mapping=new int[9][10];
	private double[][][] test_kernelMatrix=new double[45][][];
	
	private Kernel kernel;
	{//initialization
		for(int i=0;i<trainingSamples.length;i++)
			trainingSamples[i]=new ArrayList<Store>();					
		for(int i=0;i<testSamples.length;i++)
			testSamples[i]=new ArrayList<Store>();		
		for(int i=0;i<10;i++)
			baseline[i]=new int[10];
		for(int i=0;i<45;i++)
			map[i]=new int[2];
		int count=0;
		for(int x=0;x<=8;x++){			
			for(int y=x+1;y<=9;y++){
				map[count][0]=x;
				map[count][1]=y;
				mapping[x][y]=count;
				count++;
			}
		}
	}
	
	
	public void readTrs(){//train sample
		String line=null;
		String item[];
		BufferedReader reader=null;
		int count=0;
		double temp=0;
		try {
			reader = new BufferedReader(new FileReader("src/support_vector_machine_SVM/tr.csv"));			
			while((line=reader.readLine())!=null){
				item=line.split(",");
				for(int i=0;i<item.length;i++){
					temp=Double.parseDouble(item[i]);
					if(temp!=0)
						trainingSamples[count].add(new Store(i,temp));					
				}									
				count++;				
			}
		} catch (Exception e) {			
			e.printStackTrace();
		}
//		System.out.println(count);
	}
	
	public void readTss(){// test sample
		String line=null;
		String item[];
		BufferedReader reader=null;
		int count=0;
		double temp=0;
		try {
			reader = new BufferedReader(new FileReader("src/support_vector_machine_SVM/ts.csv"));
			while((line=reader.readLine())!=null){
				item=line.split(",");
				for(int i=0;i<item.length;i++){
					temp=Double.parseDouble(item[i]);
					if(temp!=0)
						testSamples[count].add(new Store(i,temp));					
				}									
				count++;				
			}
		} catch (Exception e) {			
			e.printStackTrace();
		}
	}
	
	public void readTrl(){// train label
		String line=null;		
		BufferedReader reader=null;
		int count=0;		
		try {
			reader = new BufferedReader(new FileReader("src/support_vector_machine_SVM/trl.csv"));
			while((line=reader.readLine())!=null){
				training_samples_Labels[count]=Integer.parseInt(line);
				count++;
			}
		} catch (Exception e) {			
			e.printStackTrace();
		}
	}
	
	public void readTsl(){// test label
		String line=null;		
		BufferedReader reader=null;
		int count=0;		
		int i=0;
		try {
			reader = new BufferedReader(new FileReader("src/support_vector_machine_SVM/tsl.csv"));
			while((line=reader.readLine())!=null){
				test_samples_Labels[count]=Integer.parseInt(line);								
				count++;
			}
			
		} catch (Exception e) {			
			e.printStackTrace();
		}
	}
	
	
	

	public double learned_fund_index(int k,int in){ //learning function
		 //s=sum(alph.*target.*get_kernel1(k))+b;%bug
		
		double s=0;		
		for(int i=0;i<alph_arr[in].length;i++)			
			s=s+alph_arr[in][i]*target_arr[in][i]*test_kernelMatrix[in][i][k];//kernel_func_te(k,map_index);  k:test map_index:train
		
		s=s+b_arr[in];
		return s;
	}
	
	public void testDAGSVM1(){
		int []duel;int []can=new int[2];
		int old=0,now=0,count=0,temp=0;
		double temp1,err=0;
		int []err_a=new int[10];
		for(int i=0;i<testSamples.length;i++){// i=1:length(test_samples)
			    duel=new int [10];
			    for(int j=0;j<10;j++)
			    	duel[j]=1;
			    old=0;now=9;
			    //fprintf('%d\n\n\n',i);
			    for(int j=0;j<9;j++){
			        //fprintf('value: %d ,%d, x:%d, y:%d\n',learn_func(i,mapping(old+1,new+1)),j,old,new);
			    	count=0;
			        if(learned_fund_index(i,mapping[old][now])>0)
			            duel[now]=0;
			            //old not changed
			        else{
			            duel[old]=0;
			            old=now;
			        }
			        if(j==9)
			            break;
			        
			       // getFirsttwo();// choose new opponent
			        for(int k=0;k<10;k++){
			        	if(duel[k]==1)
			        		can[count++]=k;
			        	if(count==2)
			        		break;
			        }
			        now=can[0];
			        if(now==old)//%new and old must be different
			            now=can[1];			              
			        if(now<old){//% old must <= new
			            temp=now;
			            now=old;
			            old=temp;
			        }
			    }    
			    if(old!=test_samples_Labels[i]){
			    	 baseline[test_samples_Labels[i]][old]++;
			         err++;
			         err_a[test_samples_Labels[i]]++;
			    }
			    else
			    	baseline[test_samples_Labels[i]][test_samples_Labels[i]]++;
		}
		System.out.println("DAGSVM error rate:"+err/testSamples.length);
		for(int i=0;i<10;i++)
			System.out.println("digit "+i+" error num:"+err_a[i]);	
	}
	
	public void testDAGSVM(){
		int []duel=new int[10];
		int old=0,now=0;
		double err=0;
		int []err_a=new int[10];
		for(int i=0;i<testSamples.length;i++){// i=1:length(test_samples)
			    Arrays.fill(duel,0);
			    old=0;now=9;			    
			    for(int j=0;j<9;j++){
			        
			        if(learned_fund_index(i,mapping[old][now])>0){			    	
			            duel[now]=-1;
			            //old not changed
			        }
			        else{
			            duel[old]=-1;
			            old=now;
			        }
			        if(j==9)
			            break;
			        
			       // getFirsttwo();// choose new opponent
			        for(int k=0;k<10;k++){
			        	if(duel[k]==0 && k!=old)
			        		now=k;
			        }
			        		              
			        if(now<old){// old must <= new
			        	now^=old;
			        	old^=now;
			        	now^=old;			            
			        }
			    }    
			    if(old!=test_samples_Labels[i]){
			    	 baseline[test_samples_Labels[i]][old]++;
			         err++;
			         err_a[test_samples_Labels[i]]++;
			    }
			    else
			    	baseline[test_samples_Labels[i]][test_samples_Labels[i]]++;
		}
		System.out.println("DAGSVM error rate:"+err/testSamples.length);
		for(int i=0;i<10;i++)
			System.out.println("digit "+i+" error num:"+err_a[i]);	
	}
	
	public int getLength(int i1,int i2){//for one versus one
		int length=0;
		for(int i=0;i<trainingSamples.length;i++){
			if(training_samples_Labels[i]==i1){
				length++;
			}
			else if(training_samples_Labels[i]==i2){
				length++;
			}
		}
		return length;
	}
	
	private  void getKernelMatrix(List<Store>[] training_samples,List<Store>[] testSamples,int index){ //get all dot_product
		double [][]kernelMatrix=new double[training_samples.length][testSamples.length];
		test_kernelMatrix[index]=kernelMatrix;			
		for(int i=0;i<training_samples.length;i++){			
			for(int j=0;j<testSamples.length;j++){				
				kernelMatrix[i][j]=Mathtool.kernel(Mathtool.dot_product(training_samples[i],testSamples[j]));
				//kernelMatrix[j][i]=kernelMatrix[i][j];
			}
		}
		
	}
	
	public void chooseTwo(int i1,int i2,int index){
		System.out.println(index);
		int len=getLength(i1,i2);		
		
		List<Store>[] training_samples=new ArrayList[len];		
		target_arr[index]=new int [len];
		alph_arr[index]=new double[len];
		b_arr[index]=0;
		//index_arr[num]=index;
		//b cannot be done here	
				
//			train_samples=new ArrayList[N];
		int count=0;
		for(int i=0;i<trainingSamples.length;i++){
			if(training_samples_Labels[i]==i1){
				target_arr[index][count]=1;		
				training_samples[count++]=trainingSamples[i];
			}
			else if(training_samples_Labels[i]==i2){
				target_arr[index][count]=-1;	
				training_samples[count++]=trainingSamples[i];
			}
			
		}
		getKernelMatrix(training_samples,testSamples,index);
		new trainSVM().train(len, alph_arr[index], target_arr[index], training_samples, b_arr, 0.2, 0.01, 0.01,index,kernel);
		
	}
	
	public void initialization(int category){
		readTrs();
		readTrl();
		readTss();
		readTsl();
		
		this.category=category;
		comparator_num=category*(category-1);		
		baseline=new int[category][category];
		
		b_arr=new double[comparator_num];
		alph_arr=new double[comparator_num][];//different length		
		target_arr=new int[comparator_num][];//different length			
	}
	
	public void train_oneversusone(){
		int count=0;
		for(int i=0;i<category;i++){
			for(int j=i+1;j<category;j++)
				chooseTwo(i, j,count++);			
		}
	}
	public void printbl(){
		for(int i=0;i<10;i++){
			for(int j=0;j<10;j++){
				System.out.print(baseline[i][j]+"\t");
			}
			System.out.println();
		}
	}
	
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		
		
		testSVM svm=new testSVM();
		svm.kernel=new Kernel(1);
		
		svm.initialization(10);
		svm.train_oneversusone();			
		svm.testDAGSVM();
		svm.printbl();//print baseline
	}

	
	

}
