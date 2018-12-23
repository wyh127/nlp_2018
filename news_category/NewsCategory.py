# start: 07/10/2018 
# end: 09/10/2018
# estimated time: ~6 hours

from collections import defaultdict
import pandas as pd
import xlsxwriter

# define a n-gram generating function
def get_ngrams(string, n):
    tmp = string.lower().split()
    # add start and end symbol
    new_seq = ["START" for i in range(n-1)] + tmp + ["STOP"]
    # store all n-gram in a list
    res = list()
    # representing n-gram with a tuple
    for i in range(len(new_seq)-n+1):
        temp = tuple(new_seq[i: i+n])
        res.append(temp)
    return res

if __name__ == "__main__":
	# read file
    df = pd.read_json("/Users/apple/Desktop/News_Category_Dataset.json", lines = True)
    # aggregate
    df['text'] = df.headline + " " + df.short_description

    # define a 1-gram and 2-gram dict
    unigramcounts = defaultdict(int)
    bigramcounts = defaultdict(int)
    
    # iterate through the file to create a 1-gram dict
    for val in df.text: 
        # typecaste each val to string 
        val = str(val)
        # split val 
        tokens = get_ngrams(val, 1)
        # Converts each token into lowercase 
        for i in range(len(tokens)):
            unigramcounts[tokens[i]] += 1

    # create a 2-gram dict
    for val in df.text: 
        val = str(val) 
        tokens = get_ngrams(val, 2)
        for i in range(len(tokens)):
            bigramcounts[tokens[i]] += 1

    # write into excel file
    df1 = pd.DataFrame(data=unigramcounts, index=[0])
    df1 = (df1.T)
    df1.to_excel('/Users/apple/Desktop/dict1.xlsx')

    df2 = pd.DataFrame(data=bigramcounts, index=[0])
    df2 = (df2.T)
    df2.to_excel('/Users/apple/Desktop/dict2.xlsx')













