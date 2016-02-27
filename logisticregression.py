import os
import scipy.io.wavfile
import numpy
import warnings
from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix
from scikits.talkbox.features import mfcc

#This function calculates the probability matrix which is used to calculate the accuracy and confusion matrix
def cal_probability(X_train_norm_modified, w):
    p=numpy.power(numpy.e,numpy.dot(X_train_norm_modified,w.transpose()).transpose())
    p[5,:]=1
    return p/p.sum(axis=0)
#This function initializes the variables which are used to calculate the probability which is used to calculate the accuracies and confusion matrix
def init_probability(X_train, train_norm_min, train_norm_max, w):
    X_train_norm=(X_train-train_norm_min)/(train_norm_max-train_norm_min)
    X_train_norm_modified=numpy.append(numpy.ones((len(X_train_norm),1)),X_train_norm,1)
    return cal_probability(X_train_norm_modified, w), X_train_norm_modified
#this function is used to do the 10-fold operation, calculating confusion matrix and accuracies.
def do_process(fft_features):
    folds=10#stores the number of folds
    kf = KFold(len(fft_features), folds, shuffle=True)#calculating the KFold using 10 fold for features(FFT/MFCC) calculated
    kfold_train=[]#Stores the indexes of training data obtained by doing KFold
    kfold_test=[]#Stores the indexes of test data obtained by doing KFold
    for train, test in kf:
        kfold_train.append(train)#Storing the indexes of training data obtained by doing KFold
        kfold_test.append(test)#Storing the indexes of test data obtained by doing KFold
    classcount=6#number of classes
    etazero=0.01#initial eta zero value
    lamda=0.001#initial lambda value
    loop=250#This is the repetetion count to get the weights
    accuracies=numpy.zeros((len(kfold_train),1), dtype=float)#This variable stores the accuracies of each fold
    conf_mat_1=numpy.zeros((6, 6), dtype=int)#This variable stores the confusion matrix for all the folds
    for i in range(folds):
        w=numpy.zeros((classcount,len(fft_features[0])+1),dtype=float)#This variable stores the weights. Initializing with zeros
        X_train=[]#This variable stores the training data
        X_test=[]#This variable stores the test data
        delta=numpy.zeros((classcount,len(kfold_train[i])), dtype=int)#This delta varialbe is used in calculating weights. Initializing with zeros
        orig_test_label=numpy.zeros((len(kfold_test[0]),1),dtype=int)#This variable is used to store the labels of test data
        c=0
        #Getting Training data and calculating delta matrix        
        for j in kfold_train[i]:
            X_train.append(fft_features[j])
            if j in range(0,100):
                delta[0][c]=1
            elif j in range(100,200):
                delta[1][c]=1
            elif j in range(200,300):
                delta[2][c]=1
            elif j in range(300,400):
                delta[3][c]=1
            elif j in range(400,500):
                delta[4][c]=1
            elif j in range(500,600):
                delta[5][c]=1
            c=c+1
        c1=0
        #Getting Test data and getting the labels of test data
        for k in kfold_test[i]:
            X_test.append(fft_features[k])
            if k in range(0,100):
                orig_test_label[c1]=1
            elif k in range(100,200):
                orig_test_label[c1]=2
            elif k in range(200,300):
                orig_test_label[c1]=3
            elif k in range(300,400):
                orig_test_label[c1]=4
            elif k in range(400,500):
                orig_test_label[c1]=5
            elif k in range(500,600):
                orig_test_label[c1]=6
            c1=c1+1
        train_norm_max=numpy.max(X_train, axis=0)#Getting maximum normalized value
        train_norm_min=numpy.min(X_train, axis=0)#Getting the minimum normalized value
        prob, X_train_norm_modified=init_probability(X_train, train_norm_min, train_norm_max, w)#Initializing the variables to calculate the probabiity
        #Calculating the weights by looping it 250 times        
        for m in range(loop):
            w=numpy.add(w,numpy.multiply(etazero/float(1+(float(m)/float(loop))),(numpy.subtract(numpy.dot((numpy.subtract(delta,prob)),X_train_norm_modified),numpy.multiply(lamda,w)))))
            prob=cal_probability(X_train_norm_modified, w)#Calculating the probability for the weight obtained
        prob_test, X_test_norm_modified=init_probability(X_test, train_norm_min, train_norm_max, w)#Initializing the variables to calculate the probabiity
        conf_mat=confusion_matrix(orig_test_label,(prob_test.argmax(axis=0))+1)#Calculating the confusion matrix of each genre
        conf_mat_1=conf_mat_1+ numpy.array(conf_mat)#Calculating the confusion matrix for all the genres
        sum=0.0
        for n in range(len(conf_mat)):
            sum=sum+conf_mat[n][n]
        accuracies[i]=(sum/len(X_test_norm_modified))*100#Calculating the accuracies in each fold
    print("Avg. accuracy:"+str(numpy.mean(accuracies,axis=0)))#calculating and printing Average accuracy of all the 10 folds
    print("Confusion Matrix:")
    print(conf_mat_1)#Printing the confusion matrix 
    
# Program starts here
if __name__=='__main__':
    #Defining and initializing the variables
    datasetpath="E:/Courses/Machine Learning/Project - 3/Dataset" #Path where Dataset is located. Change this to the data set path in your system
    warnings.filterwarnings("ignore")#to supress unwanted warnings that may raise while execution of program
    fft_features=[]#stores the FFT features
    mfcc_features=[]#stores the MFCC features
    per=0#this variable is used to print the percentage of program execution
    print("Reading .wav files, calculating FFT and MFCC. It may take upto 10min. to complete:    "),
#Reading the files and calculating the FFT and MFCC features for all the .wav files    
    for folder in os.listdir(datasetpath):#loop for each directory i.e., classical, country, jazz, metal, pop, rock
        file_path=datasetpath +"/"+ folder#path of each folder
        for filepath in os.listdir(file_path):#loop for each file in a folder(genre)
            filename=file_path+"/"+filepath#path of each file
            sample_rate, X= scipy.io.wavfile.read(filename)#Reading each .wav file in a genre
            fft_features.append(abs(scipy.fft(X)[:1000]))#calculating FFT, its absolute value for all the .wav files of all genres and storing the result in fft_features variable
            ceps, mspec, spec=mfcc(X)#calculating the MFCC for all the .wav files
            mfcc_features.append(numpy.mean(ceps[int(ceps.shape[0]*1/10):int(ceps.shape[0]*9/10)], axis=0))#Storing the MFCC results(features) in the mfcc_features variable
            #below 3 lines are used to print the % of program execution            
            per=per+1
            print('\b\b\b\b\b'),
            print(str(per/6)+"%"),
    fft_features=numpy.array(fft_features)#converting the fft_features list array to 2D array format
    print("")    
    print("Calculating Accuracies and Confusion matrix for FFT of 1000 features:")
    do_process(fft_features)#this function is used to do the 10-fold operation, calculating confusion matrix and accuracies.
    
    #Ranking the FFT features using the standard deviation
    print("Ranking the FFT features and calculating Accuracies and Confusion matrix for top 120 features:")
    std_genre=[]#stores the standard deviation of each genre
    std=numpy.std(fft_features, axis=0)#stores the standard deviation of all the genres
    for i in range(0,6):    
        std_genre.append(numpy.std(fft_features[(i*100):((i+1)*100),:], axis=0))
    std_index=numpy.argsort(numpy.abs(std_genre-std),axis=1)#sorting the obtained standard deviations
    top_120_index=numpy.zeros((1,120), dtype=int)#variable declaration and initialization with zeros. This is used to store top 20 features of each genre(20*6).total 120 features
    for i in range(0,6):
        top_120_index[0,(i*20):(i*20)+20]=std_index[i,0:20]##taking the top 20 ranked feature indexes of each genre and storing in the top_120_index variable
    uniq_top_120_index=numpy.unique(top_120_index)#among the 120 features taking unique feature indexes
    new_fft_features=fft_features[:,top_120_index.reshape(120)]# from the 120 indexes obtained, getting the features and storing those in new_fft_features variable
    do_process(new_fft_features)#this function is used to do the 10-fold operation, calculating confusion matrix and accuracies.
    #here we are calculating confusion matrix, accuracies using 10-fold for the MFCC features calculated at starting    
    print("Calculating Accuracies and Confusion matrix for MFCC:")
    do_process(mfcc_features)#this function is used to do the 10-fold operation, calculating confusion matrix and accuracies.