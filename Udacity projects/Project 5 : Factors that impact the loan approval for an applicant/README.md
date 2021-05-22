# Factors that impact the loan approval for an applicant

## Dataset

This [data set](https://www.google.com/url?q=https://s3.amazonaws.com/udacity-hosted-downloads/ud651/prosperLoanData.csv&sa=D&ust=1547699802003000)
contains 113,937 loans with 81 variables on each loan, including loan amount, 
borrower rate (or interest rate), current loan status, borrower income, and many others.
This [data dictionary](https://docs.google.com/spreadsheets/d/1gDyi_L4UvIrLTEC6Wri5nbaMmkGmLQBk-Yx3z0XDEtI/edit?usp=sharing) explains the 
variables in the data set.
The project objective is not expected to explore all of the variables in the dataset! But focus on only exploration on about 10-15 of them.

## Summary of Findings


1. In the loan_data dataset, there have 113937 rows and 24 columns.All the data types are correct.There have missing values in the EstimatedEffectiveYield,BorrowerAPR , 
ProsperRating (numeric),ProsperRating (Alpha) , ProsperScore , EmploymentStatus , Occupation ,EmploymentStatusDuration,DebtToIncomeRatio,BorrowerState columns.
2. LoanStatus of all Borrowers are with current and completed state.

<img src="https://github.com/rakibul-sitab/Data-Analyst-Portfolio/blob/main/Udacity%20projects/Project%205%20:%20Factors%20that%20impact%20the%20loan%20approval%20for%20an%20applicant/images/newplot%20(20).png">

3. EmploymentStatus of all Borrowers are with Employed State and most of them are full time worker.

<img src="https://github.com/rakibul-sitab/Data-Analyst-Portfolio/blob/main/Udacity%20projects/Project%205%20:%20Factors%20that%20impact%20the%20loan%20approval%20for%20an%20applicant/images/newplot%20(21).png">

4. People having middle middle income(50,000-74,999 USD) and low middle income (25,000-49,999 USD) tool more loans.Job less and low income people have less chance to get loans from bank.

<img src="https://github.com/rakibul-sitab/Data-Analyst-Portfolio/blob/main/Udacity%20projects/Project%205%20:%20Factors%20that%20impact%20the%20loan%20approval%20for%20an%20applicant/images/newplot%20(22).png">

5. Top 5 states of all Borrowers are from CA,NY,TX,FL and IL.

<img src="https://github.com/rakibul-sitab/Data-Analyst-Portfolio/blob/main/Udacity%20projects/Project%205%20:%20Factors%20that%20impact%20the%20loan%20approval%20for%20an%20applicant/images/newplot%20(23).png">

6. Most of the borrowers occupation are not defined.May be self employed like property owner.But majority are with an occupation of Professional and Executive.

<img src="https://github.com/rakibul-sitab/Data-Analyst-Portfolio/blob/main/Udacity%20projects/Project%205%20:%20Factors%20that%20impact%20the%20loan%20approval%20for%20an%20applicant/images/newplot%20(24).png">

7. Majority of the borrowers are with a rating or score from 4 to 8.They have higher chance to approve loan.

<img src="https://github.com/rakibul-sitab/Data-Analyst-Portfolio/blob/main/Udacity%20projects/Project%205%20:%20Factors%20that%20impact%20the%20loan%20approval%20for%20an%20applicant/images/newplot%20(25).png">

8. Borrower interest Rate : The average interest rate is 0.19.The maximum interest rate is 0.36 and minimum rate is 0.04.But most of the borrowers inetrest rate is 0.16.
9. Stated Monthly Income : The average monthly income is approx. 6002 USD.The maximum income is 483333 USD and minimum rate is 0.25.But most of the borrowers monthky incomw 4500   USD.
10. Loan Original Amount: The average amount is 9294 USD.The maximum loan is 35000 USD and minimum is 1000.But most of the borrowers loan amaount is approx. 4500 USD.
11. Employment status duration: The average amount is 104.5 months.The maximum is 755 months and the minimum is 0.But most of the borrowers took loan whose have employement status 0-50 months.
12. Loan original amount and monthly loan payment is highly correlated.
13. Borrower annual percentage rate and prosper score is negatively correlated.
<img src="https://github.com/rakibul-sitab/Data-Analyst-Portfolio/blob/main/Udacity%20projects/Project%205%20:%20Factors%20that%20impact%20the%20loan%20approval%20for%20an%20applicant/images/newplot%20(26).png">
14. For Applicants(employed and fulltime) with prosper ratings from 7 to 4 have the higher loan amount with increased salary.
15. For Applicants(parttime employee) with prosper ratings from 7 to 4 have the lower loan amount with low level salary.
<img src="https://github.com/rakibul-sitab/Data-Analyst-Portfolio/blob/main/Udacity%20projects/Project%205%20:%20Factors%20that%20impact%20the%20loan%20approval%20for%20an%20applicant/images/newplot%20(27).png">
16. We observe that without homeowner tend to have a higher interest rate, and thus lower rating.However homeowner tends to have lower interest rate and higher rating. So we can 
<img src="https://github.com/rakibul-sitab/Data-Analyst-Portfolio/blob/main/Udacity%20projects/Project%205%20:%20Factors%20that%20impact%20the%20loan%20approval%20for%20an%20applicant/images/newplot%20(28).png">

To summarize this report, I believe that the loan approval status is heavily influenced by the applicant's details on income range, house owner status, and job status.
