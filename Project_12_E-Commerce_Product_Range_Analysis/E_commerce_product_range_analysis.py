#!/usr/bin/env python
# coding: utf-8

# <div style="border:solid black 2px; padding: 10px">
# 
# # E-Commerce: Product Range Analysis
#     
# ## Project description :
# 
# Company XYZ have contacted with our data consulting firm.To boost their sales and targeting customers, they are interested to do a product range analysis.They want to see some KPI , such as revenue, average check, and ARPPU.

# <div style="border:solid black 2px; padding: 10px">
# 
# # Description of the data :
#     
# The dataset contains the transaction history of an online store that sells household goods.
# 
# The file 'ecommerce_dataset_us.csv' contains the following columns:
# 
# - InvoiceNo — order identifier
# 
# - StockCode — item identifier
# 
# - Description — item name
# 
# - Quantity
# 
# - InvoiceDate — order date
# 
# - UnitPrice — price per item
# 
# - CustomerID
# 

# <div style="border:solid black 2px; padding: 10px">
# 
# ## Outline
# 
# ### Data Preprocessing
# - Study missing values, type correspondence and duplicate values.
# - If necessary, remove duplicates, rename columns, convert types. 
# - Identify the product category
# 
# ### Exploratory Data Analysis
#     
# 1. How many people purchase every day, week, and month?
# 2. How many orders do cusotmer make during a given period of time? 
# 3. How much money do customer bring? (LTV)?
# 4. What is the average purchase size?
# 5. When do people start buying (purchase retention)?
# 6. What products sold the most?
# 7.   Which are the most selling category?
# 8. Recommend the most appropriate time to display advertising to maximize the likelihood of customers buying the products?
# 9. RFM Analysis
# 10. What products are most often sold together?
# 
#     
# ### Testing Hypothesis
# #### Hypothesis: Average Unit price for top selling 20 products and low selling 20 products are the different.

# <div style="border:solid black 2px; padding: 10px">
# 
# ## Conclusion
# #### Describe summary of findings and recommendation
# 

# In[3]:


import pandas as pd
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from operator import attrgetter
import plotly.express as px
import datetime as dt
from scipy import stats as st
import plotly.offline as pyoff
import plotly as py
import plotly.graph_objs as go
from sklearn.cluster import KMeans
import sklearn.cluster
from sklearn.metrics import silhouette_score
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")


# In[4]:


LOCAL_PATH_1 = 'ecommerce_dataset_us.csv'
PLATFORM_PATH_1 = '/datasets/ecommerce_dataset_us.csv'
#------------------------------------------------------
try:
    ecomm = pd.read_csv(LOCAL_PATH_1 , sep ='\t')
except:
    ecomm= pd.read_csv(PLATFORM_PATH_1, sep ='\t')
#------------------------------------------------------
display(ecomm.head())


# In[5]:


display(ecomm)


# <div style="border:solid black 2px; padding: 10px">
# 
# ## Data Preprocessing

# In[6]:


display(ecomm.info())
display(ecomm.isnull().sum())

#checking duplicates in dataframe
print('duplicate rows:',ecomm.duplicated().sum())


# <div style="border:solid black 2px; padding: 10px">
# 
# * Conclusion:
#     
# 1. There have one dataframe ecomm. The ecomm data set has 541909 rows and 7 columns.
#     
# 1. There have missing values observed in the 'Description' and 'CustomerID' columns and 5268 numbers of duplicated values have been found.

# In[7]:


# removing product which have unitprice zero
ecomm = ecomm[(ecomm['UnitPrice']>0)]
ecomm  = ecomm [(ecomm ['Quantity']>0)]


# In[8]:


#dealing with null values
missing_list=[]
for x in ecomm:
    if len(ecomm[ecomm[x].isnull()])>0:
        missing_list.append(x)
print(missing_list)

missing_perc=[]
for x in missing_list:
    missing_perc.append([x,(len(ecomm[ecomm[x].isnull()])/len(ecomm))])
missing_perc=pd.DataFrame(missing_perc,columns=['column','missing %'])
display(missing_perc.sort_values(by=['missing %'],ascending=False))


# In[9]:


ecomm.dropna(inplace =True)

#drop duplicates value
ecomm=ecomm.drop_duplicates(subset=["InvoiceNo","StockCode","Description","Quantity","InvoiceDate","UnitPrice","CustomerID"])
print('duplicate rows:',ecomm.duplicated().sum())


# In[10]:


display(ecomm.info())
display(ecomm.isnull().sum())


# In[11]:


#making the column names in lowecase
ecomm.columns = ecomm.columns.str.lower()
ecomm['description'] = ecomm['description'].str.lower()
ecomm.head()


# <div style="border:solid black 2px; padding: 10px">
# 
# * Conclusion:
#     
# 1. Missing percentage have been checked and missing values have dropped without trying to fill up because most the missing values from 'customerid' column.Duplicated rows have been removed too.
#     
# 1. To understsand easily, column names and values inside description column have been converted to lower cases.

# In[12]:


# convert datetime from object
ecomm['invoicedate']=pd.to_datetime(ecomm['invoicedate'])

# Add a separate date and time column
ecomm['invoice_date'] = ecomm['invoicedate'].dt.date
ecomm['invoice_time'] = ecomm['invoicedate'].dt.time
ecomm['invoice_month'] = ecomm['invoicedate'].dt.month
ecomm['invoice_week'] = ecomm['invoicedate'].dt.week

display(ecomm.head(1))
display(ecomm.info(1))

import warnings
warnings.filterwarnings("ignore")


# In[13]:


ecomm.rename(columns = {'customerid':'customer_id', 'unitprice':'unit_price',
                              'stockcode':'stock_code', 'invoiceno':'invoice_no'}, inplace = True)


# In[14]:


ecomm = ecomm[['customer_id','invoice_no','stock_code','quantity','unit_price','invoice_date','invoice_time','invoice_month','invoice_week','description']]


# In[15]:


ecomm.head()


# <div style="border:solid black 2px; padding: 10px">
# 
# * Conclusion:
#     
# 1. invoicedate column have been converted to datetime type,and date and time have seperated for future analysis.Column names have changed to understand easily.
#     

# In[16]:


import nltk

w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()

def lemmatize_text(text):
    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]


# In[17]:


ecomm['lemmatized']=ecomm['description'].apply(lemmatize_text)


# In[18]:


ecomm.head(5)


# In[19]:


from collections import Counter
frequencies = Counter(word for sentence in ecomm['lemmatized'] for word in sentence)
for word, frequency in frequencies.most_common(50):  # get the 50 most frequent words
    print(word, frequency)


# In[20]:


# categorize product in 25 categories using lemmatization
def product_category(value):
         stemmed_list=[stemmer.stem(i) for i in value.split(' ')]
         if 'bag' in stemmed_list:
            return 'bag and baggage products'
         elif 'heart' in  stemmed_list:
            return 'heart'
         elif 'set' in  stemmed_list:
            return 'set-box related products'
         elif 'christmas' in  stemmed_list:
            return 'christmas products'
         elif 'cake' in  stemmed_list:
            return 'cake products'
         elif 'metal' in  stemmed_list:
            return 'metal products'
         elif 'holder' in  stemmed_list:
            return 'holder products'
         elif 'paper' in  stemmed_list:
            return 'paper products'
         elif 'wooden' in  stemmed_list:
            return 'wooden products'
         elif 'lunch' in  stemmed_list:
            return 'lunch products'
         elif 'metal' in  stemmed_list:
            return 'metal products'
         elif 'water' in  stemmed_list:
            return 'water related products'
         elif 'glass' in  stemmed_list:
            return 'glass products'
         elif 'candle' in  stemmed_list:
            return 'candle products'
         elif 'ceramic' in  stemmed_list:
            return 'ceramic products'
         elif 'tea' in  stemmed_list:
            return 'tea products'
         elif 'decoration' in  stemmed_list:
            return 'decoration products'
         elif 'retrospot' in  stemmed_list:
            return 'retrospot products'
         elif 'alarm' in  stemmed_list:
            return 'alarm products'
         elif 'light' in  stemmed_list:
            return 'light products'
         elif 'spaceboy' in  stemmed_list:
            return 'spaceboy products'
         elif 'red' in  stemmed_list:
            return 'reddish products'
         elif 'box' in  stemmed_list:
            return 'box items'
         elif 'dolly' in  stemmed_list:
            return 'children items'
         elif 'pink' in  stemmed_list:
            return 'girl items'
         elif 'home' in  stemmed_list:
            return 'home items'
         else:
            return 'other'
        
ecomm['product_category']= ecomm['description'].apply(product_category)
display(ecomm.head(20))


# <div style="border:solid black 2px; padding: 10px">
# 
# * Conclusion:
#     
# Using lemmatization technique, whole products have been categorized into 25 categories.

# <div style="border:solid black 2px; padding: 10px">
# 
# ## Exploratory data analysis

# In[21]:


time_period_start_date = ecomm["invoice_date"].min()
time_period_end_date = ecomm["invoice_date"].max()
print('time_period_start_date:',time_period_start_date)
print('time_period_end_date:',time_period_end_date)


# <div style="border:solid black 2px; padding: 10px">
# 
# #### How many people purchase every day, week, and month?

# In[22]:


#How many people order over the time period?

total_people = ecomm['customer_id'].nunique()
print('total_people:', total_people)

#How many people order per day?

people_per_day = ecomm.groupby(['invoice_date']).agg({'customer_id': ['nunique']})
people_per_day.columns = ['no_of_people']
print('Average people per day:',people_per_day.mean())

#--------------------------------------------------------
#How many people order per week?

people_per_week = ecomm.groupby(['invoice_week']).agg({'customer_id': ['nunique']})
people_per_week.columns = ['no_of_people']
print('Average people per week:',people_per_week.mean())


#--------------------------------------------------------
#How many people order per month?

people_per_month = ecomm.groupby(['invoice_month']).agg({'customer_id': ['nunique']})
people_per_month.columns = ['no_of_people']
print('Average people per month:',people_per_month.mean())


# <div style="border:solid black 2px; padding: 10px">
# 
# * Conclusion:
# 
# 1. 4338 number of people purchase over the given peiod.On average, 54 people purchase per day,308 people per week and 1064 people per month.
#     
# 2. 1064 people per month, but 4338  number of people purchase throuhout the year.That means same people bought many times throuhout the year.

# <div style="border:solid black 2px; padding: 10px">
# 
# #### How many orders do customer make during a given period of time?

# In[23]:


#Total purchases by this given period?
total_order = ecomm['invoice_no'].count()
print('total_order:',total_order)


# In[24]:


#How many purchases are there per day?

order_per_day = ecomm.groupby(['invoice_date']).agg({'invoice_no': ['count']})
order_per_day.columns = ['no_of_order']
print('Average order per day:',order_per_day.mean())


# In[25]:


#How many purchases are there per week?
order_per_week = ecomm.groupby(['invoice_week']).agg({'invoice_no': ['count']})
order_per_week.columns = ['no_of_order']
print('Average order per week:',order_per_week.mean())


# In[26]:


#How many purchases are there per month?

order_per_month = ecomm.groupby(['invoice_month']).agg({'invoice_no': ['count']})
order_per_month.columns = ['no_of_order']
print('Average order per month:',order_per_month.mean())


# In[27]:


#Dynamic of purchase per day

plt.figure(figsize=[12,8])
plt.grid(True)
plt.plot(order_per_day['no_of_order'],label='order per day')
plt.title('Dynamic of order per day')
plt.xlabel("Time period")
plt.ylabel("No of order per day")
plt.show()

#----------------------------------------
#Dynamic of purchase per day

plt.figure(figsize=[12,8])
plt.grid(True)
plt.plot(order_per_week['no_of_order'],label='order per week')
plt.title('Dynamic of order per week')
plt.xlabel("Time period")
plt.ylabel("No of order per week")
plt.show()
#-------------------------------------

#Dynamic of purchase per month

plt.figure(figsize=[12,8])
plt.grid(True)
plt.plot(order_per_month['no_of_order'],label='order per month')
plt.title('Dynamic of order per month')
plt.xlabel("Time period")
plt.ylabel("No of order per month")
plt.show()


# <div style="border:solid black 2px; padding: 10px">
# 
# * Conclusion:
#     
# 1. The company shared their sales record from 2018-11-29 to 2019-12-07 time peridod.In this period, company sells total 39,2692 products.And on average they did 1287 sells per day, 7699 per week, 32724 per month.
#     
# 2. From the dynamic graphs we can see that, company sells highest number of sells between 42-49 week number.In the month of november and beginning of december they sells highest comparing to all year around.May be this sells booast is for christmas festival.

# <div style="border:solid black 2px; padding: 10px">
# 
# #### How much money do customer bring? (LTV)?

# In[28]:


ecomm_cohorts = ecomm.copy()
ecomm_cohorts['invoice_month'] = ecomm_cohorts['invoice_date'].astype('datetime64[M]')
first_orders = ecomm_cohorts.groupby('customer_id').agg({'invoice_month': 'min'}).reset_index()
first_orders.columns = ['customer_id', 'first_order_month']
orders_ = pd.merge(ecomm_cohorts,first_orders, on='customer_id')
cohorts = orders_.groupby(['invoice_month']).agg({'unit_price': 'count'}).reset_index()
cohorts.columns=['invoice_month','no_of_orders']
cohorts=cohorts.sort_values(by='invoice_month')
cohorts=cohorts.set_index('invoice_month')

display(cohorts)
display(first_orders.head())
display(orders_.head())


# In[29]:


#How much money each customer bring? (LTV)?

cohort_sizes = first_orders.groupby('first_order_month').agg({'customer_id': 'nunique'}).reset_index()
cohort_sizes.columns = ['first_order_month', 'n_buyers']
display(cohort_sizes.head())
cohorts = orders_.groupby(['first_order_month','invoice_month']).agg({'unit_price': 'sum'}).reset_index()
display(cohorts.head()) 
report = pd.merge(cohort_sizes, cohorts, on='first_order_month')
display(report.head())

margin_rate = .5 # Assuming 50% margin rate

report['gp'] = report['unit_price'] * margin_rate
report['age'] = (report['invoice_month'] - report['first_order_month']) / np.timedelta64(1, 'M')
report['age'] = report['age'].round().astype('int')
display(report.head())

report['ltv'] = report['gp'] / report['n_buyers']

output = report.pivot_table(index='first_order_month', columns='age', values='ltv', aggfunc='mean').round()
display(output.fillna(''))

ltv_201811 = output.loc['2018-11-01'].sum()
display(ltv_201811) 


# <div style="border:solid black 2px; padding: 10px">
# 
# * Conclusion:
#   
# On average, each customer from the first cohort brought USD 400 in revenue over their thirteen-month liftetime assuming 50% margin rate and considering no marketing cost.Customer life time value is approx. 400 USD.

# <div style="border:solid black 2px; padding: 10px">
# 
# #### What is the average purchase size?

# In[30]:


#What is the average purchase size?

purchase_size=  orders_.groupby(['invoice_month']).agg({'unit_price': 'mean'}).reset_index()
purchase_size.columns=['invoice_month','monthly_purchase_size']
purchase_size['invoice_month'] = pd.to_datetime(purchase_size['invoice_month']).dt.date
purchase_size=purchase_size.set_index('invoice_month')

purchase_size.plot(y='monthly_purchase_size', kind='bar', figsize=(12,8))
plt.title('Dynamic of monthly average purchase size')
plt.xlabel("Time period")
plt.ylabel("Average purchase(USD)")
plt.show()


# <div style="border:solid black 2px; padding: 10px">
# 
# * Conclusion:
#   
# On average, monthly purchase size is 3-3.5 USD all year around. Only April and May in the year 2019, it goes slide up to in between 3.5-4 Usd.

# <div style="border:solid black 2px; padding: 10px">
# 
# #### When do people start buying (purchase retention)?

# In[31]:


#When do people start buying?

order = ecomm.copy()
order = order[['customer_id','invoice_date','unit_price']]
display(order.head())

order.describe().transpose()

n_orders = order.groupby(['customer_id'])['unit_price'].nunique()
mult_orders_perc = np.sum(n_orders > 1) / order['customer_id'].nunique()
print(f'{100 * mult_orders_perc:.2f}% of customers ordered more than once.')

plt.figure(figsize=[12,8])
ax = sns.distplot(n_orders, kde=False, hist=True)
ax.set(title='Distribution of number of orders per customer',xlabel='# of orders', ylabel='# of customers')
#x.set_xlim([0, 6])
display(order.head())
average_order = n_orders.mean()
print('Average order per customer:',average_order)

import warnings
warnings.filterwarnings("ignore")


# <div style="border:solid black 2px; padding: 10px">
# 
# * Conclusion:
#   
# On average, 17 orders placed by each customer throughout the time period and 97.51% of customers ordered more than once.

# In[32]:


# convert datetime from object
order['invoice_date']=pd.to_datetime(order['invoice_date'])

order['order_month'] = order['invoice_date'].dt.to_period('M')
order['cohort'] = order.groupby('customer_id')['invoice_date'].transform('min').dt.to_period('M') 

order_cohort = order.groupby(['cohort', 'order_month']).agg(n_customers=('customer_id', 'nunique')).reset_index(drop=False)
order_cohort['period_number'] = (order_cohort.order_month - order_cohort.cohort).apply(attrgetter('n'))
cohort_pivot = order_cohort.pivot_table(index = 'cohort',columns = 'period_number',values = 'n_customers')
cohort_size = cohort_pivot.iloc[:,0]
retention_matrix = cohort_pivot.divide(cohort_size, axis = 0)
display(retention_matrix)


# In[33]:


plt.figure(figsize=(12, 8))
sns.heatmap(retention_matrix, annot=True, fmt= '.0%',cmap='YlGnBu', vmin = 0.0 , vmax = 0.6)
plt.title('Purchase retention')
plt.ylabel('Cohort Month')
plt.xlabel('Cohort Index')
plt.yticks( rotation='360')
plt.show()


# <div style="border:solid black 2px; padding: 10px">
# 
# * Conclusion:
# 
# 
# In the image, we can see that there is a sharp drop-off in the second month (indexed as 1) already, on average around 62% of customers do not make any purchase in the second month. The twelve cohort (2018–11) seems to be an exception and performs surprisingly well as compared to the other ones. A year after the first purchase, there is a 18% retention. However, from data alone, that is very hard to accurately explain.Throughout the matrix, we can see fluctuations in retention over time. This might be caused by the characteristics of the business, where clients do periodic purchases, followed by periods of inactivity.

# <div style="border:solid black 2px; padding: 10px">
# 
# #### What products sold the most? Which are the most selling category?

# In[34]:


#Top 10 products sell most

products_higest_sell = ecomm.pivot_table(index = 'description',values = 'invoice_no', aggfunc='count').reset_index()
products_higest_sell.rename(columns = {'invoice_no':'count'}, inplace = True)
products_higest_sell.sort_values(by='count',ascending=False,inplace=True)
products_higest_sell =products_higest_sell.head(10)
fig = px.bar(products_higest_sell, x='description', y='count', 
             title='Products sold most',labels={
                     "description": "Products",
                     "count": "Number of times sold"})
fig.show() 


# In[35]:


#Top 10 categories sell most

categories_higest_sell = ecomm.pivot_table(index = 'product_category',values = 'invoice_no', aggfunc='count').reset_index()
categories_higest_sell.rename(columns = {'invoice_no':'count'}, inplace = True)
categories_higest_sell.sort_values(by='count',ascending=False,inplace=True)
categories_higest_sell = categories_higest_sell.head(10)
fig = px.bar(categories_higest_sell, x='product_category', y='count', 
             title='Categories sold most',labels={
                     "product_category": "Categories",
                     "count": "Number of times sold"})
fig.show() 


# <div style="border:solid black 2px; padding: 10px">
# 
# * Conclusion:
# 
# 1. The above figures shows top 10 products and categories sold in this store.White hanging heart t-light holder and regency cake stand 3 tier are the most popular products in this store.
#     
# 2. As the products categories have been done manually using lemmatization algorithm, many products was hard to categorize with inproper name in the description.So other categories took the first place.But except the first one, bag and bagggage products, set-box related products and heart type categories products are most popular in this store. 

# <div style="border:solid black 2px; padding: 10px">
# 
# #### Recommend the most appropriate time to display advertising to maximize the likelihood of customers buying the products?

# In[36]:


#people purchase day-time

ecomm['invoice_hour'] = pd.to_datetime(ecomm['invoice_time'],format='%H:%M:%S').dt.hour
display(ecomm.head())
purchase_day_time = ecomm.groupby(['invoice_hour']).agg({'invoice_no': ['count']})
purchase_day_time.columns = ['no_of_sales']

#--------------------------------------

#people purchase week-day
ecomm['invoice_day'] = pd.to_datetime(ecomm['invoice_date']).dt.dayofweek
display(ecomm.head())
purchase_day_week = ecomm.groupby(['invoice_day']).agg({'invoice_no': ['count']})
purchase_day_week.columns = ['no_of_sales']


# In[37]:


#Dynamic of purchase per day-time

plt.figure(figsize=[12,8])
plt.grid(True)
plt.plot(purchase_day_time['no_of_sales'],label='purchase day_time')
plt.title('Dynamic of purchase day_time')
plt.xlabel("Time period")
plt.ylabel("No of purchase day_time")
plt.show()

#----------------------------------------
#Dynamic of purchase week day

plt.figure(figsize=[12,8])
plt.grid(True)
plt.plot(purchase_day_week['no_of_sales'],label='purchase week day')
plt.title('Dynamic of people purchase week day')
plt.xlabel("Time period")
plt.ylabel("No of purchase week day")
plt.show()


# <div style="border:solid black 2px; padding: 10px">
# 
# * Conclusion:
# 
# For increase sales,marketing is the key.Nowadays, digital marketing got the more popularity for the covid 19 period.If this company want to introduce digital marketing in the future, they need smart campaigns based on customer behavior. I have some recommendation for them.
#     
# 1. To optimise budget and reach maximum customers , digital marketing should be target based customer. From our analysis, we can see 12 pm is highest purchase time for this store.So, as of a few hours ago, 8am to 12 pm is the most productive time of day for conversion. As a result, the most frequent commercials should be placed during this time to boost sales.
#     
# 2. On Friday ,Saturday,and sunday the number of campaigns should be limited. Consequently, more advertisements should be placed on wednesday and Thrusday.
#     
# 3. At the beginning of this report we saw, Beginning og november to mid of december , this store sold the highhest number of products.So, the number of campaigns should be highest in this period rather than all other year around.

# <div style="border:solid black 2px; padding: 10px">
# 
# #### RFM (Recency, Frequency, Monetary) Analysis

# In[ ]:


ecomm["invoice_date"].min(), ecomm["invoice_date"].max()

PRESENT = dt.datetime(2019,12,int("0")+1)


# In[ ]:


ecomm['total_price'] = ecomm['quantity'] * ecomm['unit_price']
ecomm['invoice_date'] = pd.to_datetime(ecomm['invoice_date'])


# In[ ]:


rfm= ecomm.groupby('customer_id').agg({'invoice_date': lambda x: (PRESENT - x.max()).days,
                                        'invoice_no': 'count',
                                        'total_price': 'sum'}).reset_index()


# In[ ]:


rfm.rename(columns={'invoice_date' : 'recency',
                   'invoice_no':'frequency',
                   'total_price':'monetary_value'},inplace=True)
rfm


# In[ ]:


for i in rfm.columns:
    if i=='customer_id':
        continue
    else:
        plot_data=[
             go.Histogram(x=rfm[i])
        ]
        plot_layout=go.Layout(title=i)
        fig = go.Figure(data=plot_data,layout =plot_layout)
        pyoff.iplot(fig)


# In[ ]:


f_labels=range(1,5)
rfm['F']=pd.qcut(rfm.frequency,q=4,labels=f_labels)


# In[ ]:


rfm['F'].unique()


# In[ ]:


m_labels=range(1,5)
rfm['M']=pd.qcut(rfm.monetary_value,q=4,labels=f_labels)


# In[ ]:


rfm['M'].unique()


# In[ ]:


r_labels=range(4,0,-1)
rfm['R']=pd.qcut(rfm.recency,q=4,labels=f_labels)


# In[ ]:


rfm['R'].unique()


# In[ ]:


rfm


# In[ ]:


rfm['RFM_segment'] =rfm['R'].astype('str')+rfm['F'].astype('str')+rfm['M'].astype('str')


# In[ ]:


rfm


# In[ ]:


rfm['RFM_segment'].nunique()


# In[ ]:


rfm['RFM_segment'].value_counts()


# In[ ]:


def rfm_total(x):
    if x>=10 :
        return 'Whipped Cream'
    elif x>=9 and x<10:
        return 'Champion'
    elif x>=8 and x<9:
        return 'Loyal'
    elif x>=7 and x<8:
        return 'Potential'
    elif x>=6 and x<7:
        return 'Promising'
    elif x>=5 and x<6:
        return 'Needs attention'
    else: 
        return 'AT risk'
        


# In[ ]:


rfm['total_score']=rfm[['R','F','M']].sum(axis=1)


# In[ ]:


rfm['group']=rfm['total_score'].apply(rfm_total)


# In[ ]:


rfm


# In[ ]:


#writing function for any parameter in our rfm table
def cluster_solution(parameter):
    score ={}
    for n_cluster in [2,3,4,5,6]:
        kmeans= KMeans(n_clusters=n_cluster).fit(
            rfm[[parameter]])
        
        silhouette_avg = silhouette_score(
            rfm[[parameter]],
            kmeans.labels_)
        score[n_cluster] =silhouette_avg
    return score    
        


# In[ ]:


dict_test=cluster_solution('recency')


# In[ ]:


dict_test


# In[ ]:


#building function for any parameter in our rfm table
def cluster_solution(parameter):
    score ={}
    for n_cluster in [2,3,4,5,6]:
        kmeans= KMeans(n_clusters=n_cluster).fit(
            rfm[[parameter]])
        
        silhouette_avg = silhouette_score(
            rfm[[parameter]],
            kmeans.labels_)
        score[n_cluster] =silhouette_avg
        needed_number =max(score,key=score.get)
    return needed_number    
        


# In[ ]:


#iterating over the columns and getting back the optimal number of clusters
for i in rfm[['recency','frequency','monetary_value']].columns:
    print(i,cluster_solution(i))


# In[ ]:


rfm


# In[ ]:


writer=pd.ExcelWriter('rfm.xlsx')
rfm.to_excel(writer,'RFM')
writer.save()


# <div style="border:solid black 2px; padding: 10px">
# 
# * Conclusion:
#     
# Customers with the lowest recency, highest frequency and monetary amounts considered as top customers.From RFM analysis dashboards, we can easily filter out the potential and best customers list.And according to the below table, we can take action to get back our potentioal customers.
#     
# ![action.png](attachment:action.png)
# 
#  RFM analysis Dashboards :<https://public.tableau.com/app/profile/rakibul.islam.sitab/viz/RFManalysis_16252255667540/Dashboard1>

# <div style="border:solid black 2px; padding: 10px">
# 
# #### What products are most often sold together?

# In[38]:


sold_most_together = ecomm[ecomm['invoice_no'].duplicated(keep=False)]


sold_most_together['Grouped'] = sold_most_together.groupby('invoice_no')['description'].transform(lambda x: ','.join(x))
sold_most_together_2 = sold_most_together[['invoice_no', 'Grouped']].drop_duplicates()

import warnings
warnings.filterwarnings("ignore")


# In[39]:


from itertools import combinations
from collections import Counter

count = Counter()

for row in sold_most_together_2['Grouped']:
    row_list = row.split(',')
    count.update(Counter(combinations(row_list, 2)))

for key,value in count.most_common(10):
    print(key, value)


# <div style="border:solid black 2px; padding: 10px">
# 
# * Conclusion:
#     
# Pair with key fob , key fob with back door and key fob with shed have been often sold together and most popular.This list could be provided to the machine learning engineer for building algorithm for the recommendation products in the web store.

# <div style="border:solid black 2px; padding: 10px">
#     
# ## Testing Hypothesis

# <div style="border:solid black 2px; padding: 10px">
#     
# #### Average Unit price for top selling 20 products and low selling 20 products are the different.

# <div style="border:solid black 2px; padding: 10px">
# 
# **Testing:**
# Let's compare the sample means for average unit price.
# 1. H0  - the sample means have no difference.
# 1. H1  - the sample means are different.
# 1. alpha - 0.05

# In[40]:


#Finding top 20 average unit price

top_20_selling_products_price =ecomm.pivot_table(index = 'description',values = 'invoice_no' ,aggfunc='count').reset_index()
top_20_selling_products_price.rename(columns = {'invoice_no':'count'}, inplace = True)
top_20_selling_products_price.sort_values(by='count',ascending=False,inplace=True)
top_20_selling_products_price_1 = top_20_selling_products_price.head(20)
display(top_20_selling_products_price_1)


# In[41]:


result_1 = pd.merge(top_20_selling_products_price_1,
                 ecomm[['description','unit_price']],
                 on='description')


# In[42]:


result_1.head()


# In[43]:


top_20_products = result_1.pivot_table(index = 'description',values = 'unit_price' ,aggfunc='mean').reset_index()
top_20_products.rename(columns = {'unit_price':'average_unit_price'}, inplace = True)
top_20_products.sort_values(by='average_unit_price',ascending=False,inplace=True)


# In[44]:


top_20_products


# In[45]:


#Finding low 20 average unit price

low_20_selling_products_price = top_20_selling_products_price.tail(20)


# In[46]:


result_2 = pd.merge(low_20_selling_products_price,
                 ecomm[['description','unit_price']],
                 on='description')


# In[47]:


result_2.head()


# In[48]:


low_20_products = result_2.pivot_table(index = 'description',values = 'unit_price' ,aggfunc='mean').reset_index()
low_20_products.rename(columns = {'unit_price':'average_unit_price'}, inplace = True)
low_20_products.sort_values(by='average_unit_price',ascending=False,inplace=True)


# In[49]:


low_20_products


# In[50]:


top20_Avg_unit_price=top_20_products['average_unit_price']
low20_Avg_unit_price=low_20_products['average_unit_price']


alpha = .05   #critical statistical significance level

results = st.ttest_ind(
        top20_Avg_unit_price, 
        low20_Avg_unit_price,equal_var = False) 

print('p-value:',results.pvalue) 

if (results.pvalue < alpha):
    print("We reject the null hypothesis")
else:
    print("We can't reject the null hypothesis")


# <div style="border:solid black 2px; padding: 10px">
# 
# * Conclusion:
# 
# The analysis suggested that Average unit price for the top20 products and low20 products are not different.In both cases, a significance level of 0.05 selected which concludes a 5% risk of a difference exists when there is no actual difference. Lower significance levels indicate that i require stronger evidence before i reject the null hypothesis.

# <div style="border:solid black 2px; padding: 10px">
# 
# ### Overall Conclusion:
#     
# 1. There have one dataframe ecomm. The ecomm data set has 541909 rows and 7 columns.
#     
# 1. There have missing values observed in the 'Description' and 'CustomerID' columns and 5268 numbers of duplicated values have been found.
#     
# 1. Missing percentage have been checked and missing values have dropped without trying to fill up because most the missing values from 'customerid' column.Duplicated rows have been removed too.
#     
# 1. To understsand easily, column names and values inside description column have been converted to lower cases.
#     
# 1. Using lemmatization technique, whole products have been categorized into 25 categories.
#     
# 1. 4372 number of people purchase over the given peiod.On average, 63 people purchase per day,343 people per week and 1115 people per month.
# 
# 1. 1115 people per month, but 4372 number of people purchase throuhout the year.That means same people bought many times throuhout the year.
#     
# 1. The company shared their sales record from 2018-11-29 to 2019-12-07 time peridod.In this period, company sells total 401604 products.And on average they did 1316 sells per day, 7874 per week, 33467 per month.
# 
# 1. From the dynamic graphs we can see that, company sells highest number of sells between 42-49 week number.In the month of november and beginning of december they sells highest comparing to all year around.May be this sells booast is for christmas festival.
#     
# 1. On average, each customer from the first cohort brought USD 400 in revenue over their thirteen-month liftetime assuming 50% margin rate and considering no marketing cost.Customer life time value is approx. 400 USD.
#     
# 1. On average, monthly purchase size is 3-3.5 USD all year around. Only April and May in the year 2019, it goes slide up to in between 3.5-4 Usd.
#     
# 1. On average, 17 orders placed by each customer throughout the time period and 97.51% of customers ordered more than once.
#     
# 1. From purchase retention matrix, we can see that there is a sharp drop-off in the second month (indexed as 1) already, on average around 62% of customers do not make any purchase in the second month. The twelve cohort (2018–11) seems to be an exception and performs surprisingly well as compared to the other ones. A year after the first purchase, there is a 18% retention. However, from data alone, that is very hard to accurately explain.Throughout the matrix, we can see fluctuations in retention over time. This might be caused by the characteristics of the business, where clients do periodic purchases, followed by periods of inactivity.
#     
# 1. White hanging heart t-light holder and regency cake stand 3 tier are the most popular products in this store.
# 
# 1. As the products categories have been done manually using lemmatization algorithm, many products was hard to categorize with inproper name in the description.So other categories took the first place.But except the first one, bag and bagggage products, set-box related products and heart type categories products are most popular in this store.
#     
# 1. From RFM analysis, we have found out our top/ best customer list.
#     
# 1. Pair with key fob , key fob with back door and key fob with shed have been often sold together and most popular.This list could be provided to the machine learning engineer for building algorithm for the recommendation products in the web store.
#     
# 1. The analysis suggested that Average unit price for the top20 products and low20 products are not different.In both cases, a significance level of 0.05 selected which concludes a 5% risk of a difference exists when there is no actual difference. Lower significance levels indicate that i require stronger evidence before i reject the null hypothesis.

# <div style="border:solid black 2px; padding: 10px">
# 
# ### Future recommendation:
#     
# For increase sales,marketing is the key.Nowadays, digital marketing got the more popularity for the covid 19 period.If this company want to introduce digital marketing in the future, they need smart campaigns based on customer behavior. I have some recommendation for them.
#     
# 1. To optimise budget and reach maximum customers , digital marketing should be target based customer. From our analysis, we can see 12 pm is highest purchase time for this store.So, as of a few hours ago, 8am to 12 pm is the most productive time of day for conversion. As a result, the most frequent commercials should be placed during this time to boost sales.
#     
# 2. On Friday ,Saturday,and sunday the number of campaigns should be limited. Consequently, more advertisements should be placed on wednesday and Thrusday.
#     
# 3. At the beginning of this report we saw, Beginning og november to mid of december , this store sold the highhest number of products.So, the number of campaigns should be higher in this period rather than all other year around.
