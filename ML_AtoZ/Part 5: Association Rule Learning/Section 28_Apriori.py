"""
Machine Learning A-Z: Hands-On Python & R In Data Science
created by Kirill Eremenko, Hadelin de Ponteves, SuperDataScience Team
https://www.udemy.com/machinelearning/

Part 5: Association Rule Learning
Section 28: Apriori Algorithm

Created on Tue Apr  15, 2019
@author: yinka_ola
"""

#Data Scenario: 
## A store in the south of France
## Investigate the product sales in the store to study how to optimize sales
## Investigate how to optimize product placement and pairings to optimize sales
## 7500 transaction per week
## each customer shops at least once per week
## figure out which combination of products that customers are likely to buy


## Apriori: association rules
## general idea: People who bought also bought 
## Marketing Strategy used by grocery store
## I.e: milk and bread are placed on opposite ends of store, so that customers can walk
## around more to pick up more items
## placing cereal close to the milk to optimise sales

## Appl: recommmendation systems  (i.e Netflix, Amazon)
## if they like movie 1, they will like movie 2
## "we think you might like this"

## Apriori algorithm: using movie recommendation
## support: No user watchlist for mov 1)/ total watchlist
## confidence: No user watchlist for mov 1 and mov 2/No of wactchlist for mov 1 
## lift: confidence/support => improvement in prediction

## goal: can we recomend based on prior (apriori) knowledge

## How to build Apriori Algorithm:
## step 1: set a min support and confidence
## Step 2: take all subset in transaction w/ higher support than min support
## step 3: take all rules in transaction w/ higher support than min confidence
## step 4: sort the rules be decreasing lift
## done!


# Apriori

# Importing the libraries
import pandas as pd #data
import numpy as np #mathematics
import os
#plotting packages
import matplotlib.pyplot as plt #plotting charts
import seaborn as sns
sns.set()
%matplotlib inline
plt.rcParams['figure.figsize'] = 10,10
#ignore warnings
import warnings
warnings.filterwarnings('ignore')


# Data Preprocessing
# there is no header in this dataset
# here the apriori is expecting that each transaction is a list
# we need to prep the data to make it a list of lists
# 2 loops: 1 over transaction, the other over the dataset
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])

# Training Apriori on the dataset
# here we are using the spreadsheet from our library
# the model takes the transaction as input and association rules as output
# lets find products purchase min 3-4x per day (keep in mind the season)
# In a week - 3* 7days = 21
# min_support = 21/7500 = 0.0028 = 0.003
# min_confidence = 20 % = 0.2
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

# Visualising the results
# the rules here are aleady sorted by their relevance
# same results still obtained in R
results = list(rules)

print(results[0])  
## Observation:

## The first item in the list is a list itself containing three items. 
## The first item of the list shows the grocery items in the rule.

## For instance from the first item, we can see that light cream and chicken are commonly bought together. 
## This makes sense since people who purchase light cream are careful about what they eat hence they are more likely to buy chicken 
## white meat vs. red meat i.e. beef. 
## perhaps light cream is commonly used in recipes for chicken?

## The support value for the first rule is 0.0045. 
#This number is calculated by dividing the number of transactions containing light cream divided by total number of transactions. 
# The confidence level for the rule is 0.2905 
# which shows that out of all the transactions that contain light cream, 
# 29.05% of the transactions also contain chicken. 
# Finally, the lift of 4.84 tells us that chicken is 4.84 times more likely to be bought by the customers who buy light cream compared to the default likelihood of the sale of chicken.


## The following script displays the rule, the support, the confidence, and lift for each rule in a more clear way:

for item in results:

    # first index of the inner list
    # Contains base item and add item
    pair = item[0] 
    items = [x for x in pair]
    print("Rule: " + items[0] + " -> " + items[1])

    #second index of the inner list
    print("Support: " + str(item[1]))

    #third index of the list located at 0th
    #of the third index of the inner list

    print("Confidence: " + str(item[2][0][2]))
    print("Lift: " + str(item[2][0][3]))
    print("=====================================")
    
"""
We have already discussed the first rule. Let's now discuss the second rule. The second rule states that mushroom cream sauce and escalope are bought frequently. The support for mushroom cream sauce is 0.0057. The confidence for this rule is 0.3006 which means that out of all the transactions containing mushroom, 30.06% of the transactions are likely to contain escalope as well. Finally, lift of 3.79 shows that the escalope is 3.79 more likely to be bought by the customers that buy mushroom cream sauce, compared to its default sale.

Conclusion
Association rule mining algorithms such as Apriori are very useful for finding simple associations between our data items. They are easy to implement and have high explain-ability. However for more advanced insights, such those used by Google or Amazon etc., more complex algorithms, such as recommender systems, are used. However, you can probably see that this method is a very simple way to get basic associations if that's all your use-case needs.
"""