import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn import  linear_model
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error


#%%
def datasets(Target):
    # Read dataset with all the features google trends keywords and y target
    df  =  pd.read_csv("Datasets/"+Target+"_prevalence_diabetes_BFRSS_union_GT.csv",sep  =  ',',encoding  =  "utf-8-sig",index_col  =  0)
    # Take states
    states  =  sorted( list( set( df.State)))
    # Take start year
    start_year  =  min(df.Year)
    # Set the multiple index with the primary key (Year,State)
    df.set_index(pd.MultiIndex.from_tuples(list(zip(df.Year.astype(str),df.State))),inplace  =  True)
    df.drop(["State","Year"],axis  =  1,inplace  =  True)
    
    # Read dataset with info (CI and SampleSize) about y target
    df_info  =  pd.read_csv("Datasets/"+Target+"_prevalence_INFO_diabetes_BFRSS_union_GT.csv",sep = ',',encoding = "utf-8-sig",index_col  =  0)
    # Set the multiple index with the primary key (Year,State)
    df_info.set_index(pd.MultiIndex.from_tuples(list(zip(df_info.Year.astype(str),df_info.State))),inplace  =  True)
    df_info.drop(["State","Year"],axis  =  1,inplace  =  True)
    
    # Read data about percentage of poverty by states
    df_poverty =  pd.read_csv("Datasets/Poverty_2004_2016_AllStates.csv",sep = ',',encoding = "utf-8-sig",index_col  =  0)
    # Set the multiple index with the primary key (Year,State)
    df_poverty.set_index(pd.MultiIndex.from_tuples(list(zip(df_poverty.Year.astype(str),df_poverty.State))),inplace  =  True)
    df_poverty.drop(["State","Year"],axis  =  1,inplace  =  True)
    
    # Read data about percentage with Income less tha 15000$ by states
    df_income =  pd.read_csv("Datasets/Income_2004_2016_AllStates.csv",sep = ',',encoding = "utf-8-sig",index_col  =  0)
    # Set the multiple index with the primary key (Year,State)
    df_income.set_index(pd.MultiIndex.from_tuples(list(zip(df_income.Year.astype(str),df_income.State))),inplace  =  True)
    df_income.drop(["State","Year"],axis  =  1,inplace  =  True)
    
    # Read data about coverage percetange of Insurance by states
    df_insurance =  pd.read_csv("Datasets/Insurance_2004_2016_AllStates.csv",sep = ',',encoding = "utf-8-sig",index_col  =  0)
    # Set the multiple index with the primary key (Year,State)
    df_insurance.set_index(pd.MultiIndex.from_tuples(list(zip(df_insurance.Year.astype(str),df_insurance.State))),inplace  =  True)
    df_insurance.drop(["State","Year"],axis  =  1,inplace  =  True)
    
    # Merge datasets of Poverty, Income and  Insurance
    df_Pov_Inc  =  pd.merge(df_poverty, df_income, left_index  =  True, right_index  =  True)
    df_Pov_Inc_Ins  =  pd.merge(df_Pov_Inc,df_insurance , left_index  =  True, right_index  =  True)
    
    # Take years
    years  =  list(range(start_year,2017))
    # Take number of states
    n  =  len(states)
    # Take number of years
    t  =  len(years) # number of years  
    return (df,df_info,df_Pov_Inc_Ins,states,start_year,years,n,t)

# This function give you a formatted dataset of Google trends keywords given a path of a myfolder
def GT_KW(states,percorso,start_year):
    df_gt  =  pd.DataFrame({"State" : states})
    nm  =  0
    for y in range(start_year,2017):
        # Range(1,6) because you can take maximum 5 keywords at a time
        for i in range(1,6):
            name = percorso+str(y)+'.csv'
            # Order the states
            dataframe = pd.read_csv(name,sep = ',').sort_values(["geoName"])
            dataframe.reset_index(inplace = True,drop = True)
            # Check if I miss some states due to little amount of queries
            if len(states)!= len(dataframe['geoName']):
                out = list(set(states).difference(dataframe['geoName']))
                num = sorted([states.index(o) for o in out])
                start = 0
                idxs = []
                for n in num:
                    idxs.append(df_gt.index[start:n])
                    start = n
                index = list (set(df_gt.index.values).difference(set(num)))
                dataframe.set_index( np.array(index) ,inplace = True)
                for n in num:
                    # Replace with all zeros the missing row/state
                    dataframe.loc[n] =  [out, 0 ,0, 0 ,0,0]
                dataframe.sort_index(inplace = True)
            df_gt.insert(i+i*nm, dataframe.columns[i].replace(" ", "")+':'+str(y), dataframe[dataframe.columns[i]])
        nm+= 1
    keywords = list(set(df_gt.columns.str.split(':').str[0][1:]))
    df_gt.set_index("State",inplace = True)
    return df_gt,keywords


# Merge all google trend datasets, 5 columns at a time
def gt_dataset(per,states,start_year):
    dataset = pd.DataFrame()
    kws = []
    for p in per:
        df_gt,keywords = GT_KW(states,p,start_year)
        dataset  =  pd.concat([dataset,df_gt],axis = 1)
        kws+= keywords
    return(dataset,kws)

# Correlation across single word in all the states and the Brfss value of Diabetes, by year
def corr_matrix_by_years(df_scaler,keywords,m,t,years,states):
	Corr_kw_by_years  =  np.zeros((m,t))
	for i,y in enumerate(years):
		idxs = get_index(states,y)
		kw_across_states = df_scaler.loc[idxs]
		for j,k in enumerate(keywords):
			Corr_kw_by_years[j,i] = kw_across_states[k].corr(kw_across_states['y'])
	# Means of the correlation by years		
	corr_kw = np.mean(Corr_kw_by_years,axis = 1)
	list_kw_by_years = pd.DataFrame({ "Keyword": [x for x,y in sorted(zip(keywords,abs(corr_kw)),key  =  lambda x : x[1],reverse = True)],
                                     "Corr" : [y for x,y in sorted(zip(keywords,abs(corr_kw)),key  =  lambda x : x[1],reverse = True)] })
	list_kw_by_years = list_kw_by_years[["Keyword","Corr"]]
	return (Corr_kw_by_years,list_kw_by_years)
	
def plot_stability(Corr_kw_by_years,keywords,years,m):
	plt.style.use('fivethirtyeight')
	fig, ax  =  plt.subplots( figsize = (8, 8))
	# These are the colors that will be used in the plot
	color_sequence  =  ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c',
					  '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5',
					  '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f',
					  '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5']
	[ax.plot(years,Corr_kw_by_years[i],color_sequence[i]) for i in range(m)]
	plt.ylim([-1.3,1.3])
	ax.legend(keywords,loc = "best",fontsize = 16,bbox_to_anchor = (1, 0.75), ncol = 1)
	ax.set_ylabel("Correlation")
	ax.set_xticks(years)
	ax.set_xticklabels(years)	 
	ax.set_xlabel("Year")
	ax.set_title(" Correlation across single word in all the states and the BRFSS value by year")
	plt.show()
	
# Correlation across single word in all the years and the BFRSS value, by states
def corr_matrix_states_words(df_scaler,keywords,n,m,states,years):   
    Covar_st_wr  =  np.zeros((n,m))
    for num,st in enumerate(states):
        idxs = get_index(st,years)
        kw_across_years = df_scaler.loc[idxs]
        for i,kw in enumerate(keywords):
            Covar_st_wr[num,i] = np.nan_to_num(kw_across_years[kw].corr(kw_across_years["y"]))
    # Means of cooreleation by year         
    corr_kw_states = np.mean(Covar_st_wr,axis = 1)
    list_kw_by_states = pd.DataFrame({ "Keyword": [x for x,y in sorted(zip(keywords,abs(corr_kw_states)),key  =  lambda x : x[1],reverse = True)],
                                      "Corr" : [y for x,y in sorted(zip(keywords,abs(corr_kw_states)),key  =  lambda x : x[1],reverse = True)] })
    list_kw_by_states = list_kw_by_states[["Keyword","Corr"]]
    return(Covar_st_wr,list_kw_by_states)
	
def plot_corr_by_states(Covar_st_wr,keywords,m,level = 0.6):
	plt.rcdefaults()
	plt.style.use('fivethirtyeight')
	fig, ax  =  plt.subplots()
	y_pos  =  np.arange(m)
	performance  =  np.sum(abs(Covar_st_wr)> level,axis = 0)
	color_sequence  =  ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c',
					  '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5',
					  '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f',
					  '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5']
	ax.barh(y_pos, performance, align = 'center', color = color_sequence[:m], ecolor = 'black')
	ax.set_yticks(y_pos)
	ax.set_yticklabels((keywords),fontsize = 10)
	ax.invert_yaxis()  # labels read top-to-bottom
	ax.set_xlabel('# states')
	ax.set_title('Number of states where a specific word have a correlation greater than '+str(level))
	plt.show()
    
# Get multiindex as tuple
def get_index(s,y): # you can give a single state and number or a list of them
    if type(s) == str:
        s = [s]
    n = len(s)
    if type(y)!= list:
        y = [y,y]
        years = np.repeat(list(map(str,range(y[0],y[1]+1))),n)
    else:
        years = np.repeat(list(map(str,y)),n)
    t = len(years)
    sts = s*t
    idx = list(zip(years,sts))
    return idx

# Feature selection by Partial F-Test
def feature_selection(train,test,list_kw_by_years):
	x_train,y_train = train.drop(["y"],axis = 1) ,train['y']
	x_test,y_test = test.drop(["y"],axis = 1) ,test['y']
	# Create linear regression object
	regr  =  linear_model.LinearRegression() 
	# Train the model using the training sets
	regr.fit(x_train,y_train)
	# Make predictions using the training set
	diabetes_y_pred  =  regr.predict(x_train)
	RSS_full =  sum((y_train - diabetes_y_pred)**2)
	print('RSS_full : ',RSS_full)
	print('Keywords of full model :',list_kw_by_years.Keyword.tolist())
	print()
	# Value of the last (because the second degree of freedom is always larger) row of Table F at 0.05 level of significance
	F_table = [3.84,3,2.60 ,2.37 ,2.21 ,2.10 ,2.01, 1.94 ,1.88, 1.83 ,1.75,1.67 ,1.57]
	m = list_kw_by_years.shape[0]
	removed_kws=[]
	for i in range(1,m):
		k_to_remove = ["y"] +list_kw_by_years.Keyword[-i:].tolist()
		k_under_test=[k_to_remove[1]]
		print('I\'m trying to remove :\n',*k_to_remove[1:])
		x_train,y_train = train.drop(k_to_remove,axis = 1) ,train['y']
		# Create linear regression object
		regr  =  linear_model.LinearRegression() 
		# Train the model using the training sets
		regr.fit(x_train,y_train)
		# Make predictions using the training set
		diabetes_y_pred  =  regr.predict(x_train)
		RSS_reduced =  sum((y_train - diabetes_y_pred)**2)
		print('RSS_reduced : ',RSS_reduced)
		p = len(k_to_remove)-1
		k = m+1 #vars of full model + intercept
		n_sample = x_train.shape[0]+x_test.shape[0]
		
		F =  ( (RSS_reduced - RSS_full)*(1/p) ) / ( RSS_full / (n_sample-k-1))
		print("F statistic : %.3f"%F)
		df1  =  p
		df2  =  n_sample-k  #-> 3.84 3. 2.60 2.37 2.21 2.10 2.01
		print("The degrees of F are %i and %i"%(df1,df2))
		print("We need to check if F is greater than",F_table[df1])
		print()
		if F >=  F_table[df1]:
			print("We have done!")
			final_kws  =  x_train.columns.tolist()
			print("The final keywords :",final_kws)
			print("We removed :",*removed_kws)
			print()
			return(removed_kws)
		removed_kws.extend(k_under_test)
	return(list_kw_by_years.Keyword.tolist())
	
    
# Grid search with kfold a given lists of parameters
def grid_search_CV_for_years(first_grid,second_grid,df,df_scaler,df_info,years,states,num_kfold):
    FirstPar_grid =  first_grid
    SecondPar_grid =  second_grid
    
    max_error = 10000
    error = np.zeros(( len(FirstPar_grid),len(SecondPar_grid)))
    ncomb = len(FirstPar_grid)*len(SecondPar_grid)
    num_comb = 1
    for i,c1 in enumerate(FirstPar_grid):
        for j,c2 in enumerate(SecondPar_grid):
            print("--------------Combination number %i of %i ------------------------"%(num_comb,ncomb))
            print("The values C   =  %i  gamma  =  %.5f "%(c1,c2))
            num_comb+= 1
            MSE_vec = []
            R2_vec = []
            kf  =  KFold(n_splits  =  num_kfold,shuffle = True)
            for train_index, validation_index in kf.split(years):  #some index is used like test
                # PREDICTION YEARs
                idx_train = get_index(states, [years[i] for i in train_index])
                idx_validation = get_index(states, [years[i] for i in validation_index])
                
                train,validation = df_scaler.loc[idx_train],df_scaler.loc[idx_validation]
                
                x_train,y_train = train.drop(["y"],axis = 1) ,train['y']
                x_validation,y_validation = validation.drop(["y"],axis = 1) ,validation['y']
                
                # Support Vector Regression Model
                regr  =  SVR(C = c1, gamma = c2, epsilon = 0.1,tol = 1e-6) 
                # Train the model using the training sets
                regr.fit(x_train,y_train)
                # Make predictions using the testing set
                diabetes_y_pred  =  regr.predict(x_validation)
                # The mean squared error
                MSE  =  mean_squared_error(y_validation, diabetes_y_pred)
                MSE_vec.append(MSE)
                # Explained variance score: 1 is perfect prediction
                R2  =  regr.score(x_train,y_train)
                R2_vec.append(R2)
                # Rescale y_pred for CI informations
                y_pred  =  diabetes_y_pred*df.y.std() + df.y.mean()
                low  =  df_info.loc[idx_validation].Lower_CI
                upp  =  df_info.loc[idx_validation].Upper_CI
                n_sample = x_validation.shape[0]
                CI_fallen = sum((low <= y_pred) & (y_pred <= upp))
                               
                print("MSE  =  %.2f R2  =  %.2f and %i (on %i) value predicted are inside the real CI)"%(MSE,R2,CI_fallen,n_sample))
            print("--------------------------------------------")
            print()
            error[i,j] = np.mean(MSE_vec)
            
            if error[i,j] < max_error:
                max_error = error[i,j]
                list_best = [c1,c2, np.mean(MSE_vec),np.mean(R2_vec)]
                
    print("The best values C   =  %i, gamma  =  %.5f, MSE  =  %.2f and R2  =  %.2f "%(list_best[0],list_best[1],list_best[2],list_best[3]))
    return list_best
	
def grid_search_for_states(first_grid,second_grid,df,df_scaler,df_info,years,states,num_kfold):
    FirstPar_grid =  first_grid
    SecondPar_grid =  second_grid
    
    max_error = 10000
    error = np.zeros(( len(FirstPar_grid),len(SecondPar_grid)))
    ncomb = len(FirstPar_grid)*len(SecondPar_grid)
    num_comb = 1
    for i,c1 in enumerate(FirstPar_grid):
        for j,c2 in enumerate(SecondPar_grid):
            print("--------------Combination number %i of %i ------------------------"%(num_comb,ncomb))
            print("The values C   =  %i  gamma  =  %.5f "%(c1,c2))
            num_comb+= 1
            MSE_vec  =  []
            R2_vec  =  []
            kf  =  KFold(n_splits = num_kfold,shuffle = True)
            
            for train_index, validation_index in kf.split(states):  #some index is used like test
				
                # PREDICTION STATEs 
                idx_train = get_index([states [i] for i in train_index], years)
                idx_validation = get_index([states [i] for i in validation_index], years)
                
                train,validation = df_scaler.loc[idx_train],df_scaler.loc[idx_validation]
                
                x_train,y_train = train.drop(["y"],axis = 1) ,train['y']
                x_validation,y_validation = validation.drop(["y"],axis = 1) ,validation['y']
                
                # Support Vector Regression Model
                regr  =  SVR(C = c1, gamma = c2, epsilon = 0.1,tol = 1e-6) 
                # Train the model using the training sets
                regr.fit(x_train,y_train)
                # Make predictions using the testing set
                diabetes_y_pred  =  regr.predict(x_validation)
                # The mean squared error
                MSE  =  mean_squared_error(y_validation, diabetes_y_pred)
                MSE_vec.append(MSE)
                # Explained variance score: 1 is perfect prediction
                R2  =  regr.score(x_train,y_train)
                R2_vec.append(R2)
                # Rescale y_pred for CI informations
                y_pred  =  diabetes_y_pred*df.y.std() + df.y.mean()
                low  =  df_info.loc[idx_validation].Lower_CI
                upp  =  df_info.loc[idx_validation].Upper_CI
                n_sample  =  x_validation.shape[0]
                CI_fallen  =  sum((low <= y_pred) & (y_pred <= upp))
                               
                print("MSE  =  %.2f R2  =  %.2f and %i (on %i) value predicted are inside the real CI)"%(MSE,R2,CI_fallen,n_sample))
            print("--------------------------------------------")
            print()
            error[i,j] = np.mean(MSE_vec)
            
            if error[i,j] < max_error:
                max_error = error[i,j]
                list_best = [c1,c2, np.mean(MSE_vec),np.mean(R2_vec)]
                
    print("The best values C   =  %i, gamma  =  %.5f, MSE  =  %.2f and R2  =  %.2f "%(list_best[0],list_best[1],list_best[2],list_best[3]))
    return list_best

def plot_pred(y_pred_list,df_pred_years,states,years,t_pred):

	fig, ax = plt.subplots(figsize=(10, 6))
	#print(y_pred_list)
	idxs = get_index(states,years)
	y_true = df_pred_years.loc[idxs].y
	y_true_median = y_true.groupby(np.arange(len(y_true))//len(states)).median()

	list_type=["Y pred","Y pred + Income and Poverty","Y pred + Income, Poverty and Insurance"]
	color_sequence  =  ['royalblue', 'limegreen', 'peru']
	ax.plot(years,y_true_median, 'crimson', linewidth=3, label='BFRSS Value')
	for i,yp in enumerate(y_pred_list):
		idxs1= get_index(states,years[-t_pred:])
		
		y_pred = pd.DataFrame({'y_pred':yp},index=idxs1).y_pred
		y_pred_median=y_pred.groupby(np.arange(len(y_pred))//len(states)).median()
		ax.plot(years[-t_pred:],y_pred_median,color_sequence[i], linewidth=1.5, label=list_type[i])

	ax.set_xlim(years[0]-1,2017)
	ax.set_ylim(0,20)
	ax.set_xticks(years)
	ax.set_xticklabels(years,rotation=45)
	# tidy up the figure
	ax.grid(True)
	ax.legend(loc='best',fontsize=14)
	ax.set_title('BFRSS diabetes data and prediction')
	ax.set_xlabel('years')
	ax.set_ylabel('y value')

	plt.show()