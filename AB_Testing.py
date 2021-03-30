##########################################
# A/B TEST PROJECT
#########################################
# The Story
# One big company has changed the way of the advertising tenders which is maximum bidding to average bidding.
# Then our company has decided to test this new feature and wants to run an A / B test to see if average bidding yields
# more conversions than the maximum bidding.
# Target group is divided into two equal size, as test and control.
# Our main metrics are purchase and earnings and we are focusing that two variables.

# Data Sets:
# Impression: User see a advertising
# Click: User click the advertising
# Purchase: Number of purchases
# Earning:


import pandas as pd
import numpy as np
from statsmodels.stats.proportion import proportions_ztest
from scipy.stats import shapiro
from scipy.stats.stats import pearsonr
from scipy.stats import stats
import statsmodels.stats.api as sms
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from helpers.myfunc import check_data

# Control and Test Data Set

pd.set_option('display.max_columns', None)

df_c = pd.read_excel('dataset/ab_testing_data.xlsx', sheet_name='Control Group')
control = df_c.copy()
df_t = pd.read_excel('dataset/ab_testing_data.xlsx', sheet_name='Test Group')
test = df_t.copy()

control.head()
test.head()

control.shape
test.shape

control.info()
test.info()

control.describe().T
test.describe().T

# confidence interval for purchase
sms.DescrStatsW(control["Purchase"]).tconfint_mean()
sms.DescrStatsW(test["Purchase"]).tconfint_mean()

# looking kde plot
sns.kdeplot(control.Purchase, shade=True)
plt.show()

sns.kdeplot(test.Purchase, shade=True)
plt.show()

# check outlier with boxplot
sns.boxplot(x=control['Purchase'])
plt.show()

sns.boxplot(x=test['Purchase'])
plt.show()

# organizing data for AB Test
A = control[['Purchase']]
B = test[['Purchase']]
AB = pd.concat([A, B], axis=1)
AB.columns = ['C_Purchase', 'T_Purchase']
AB.head()

AB.mean()
sns.kdeplot(data=AB, shade=True)
plt.show()

# sns.histplot(data=AB, bins=25)
# plt.show()

# looking box plot graph's two data
group_a = pd.DataFrame(np.arange(len(A)))
group_a[:] = 'A'
AG = pd.concat([A, group_a], axis=1)
group_b = pd.DataFrame(np.arange(len(B)))
group_b[:] = 'B'
BG = pd.concat([B, group_b], axis=1)
A_B = pd.concat([AG, BG])
A_B.columns = ['Purchase', 'Groups']
sns.boxplot(x='Groups', y='Purchase', data=A_B)
plt.show()




#######################################
# Create hypothesis (Purchase)
#######################################
# Control and Test Purchase mean:
# C_Purchase:    550.894059
# T_Purchase:    582.106097

# H0: M1 = M2 There is no statistically significant difference between the maximum bidding and average bidding.
# H1: M1 != M2 There is a statistically significant difference between the maximum bidding and average bidding.

# Assumptions of normality
# H0 = Normal distribution assumption is provided.
# H1 = Normal distribution assumption is not provided.

test_statistics, pvalue = shapiro(AB['C_Purchase'])
print('Test statistics = %.4f, p-value = %.4f' % (test_statistics, pvalue))

#Test statistics = 0.9773, p-value = 0.5891
# h0 isn't rejected.

test_statistics, pvalue = shapiro(AB['T_Purchase'])
print('Test statistics = %.4f, p-value = %.4f' % (test_statistics, pvalue))

# Test statistics =0.9589, p-value = 0.1541
# h0 isn't rejected.

###############################################
# Variance homogeneity
# H0 = Variance is homogeneous
# H1 = Variance is not homogeneous.

stats.levene(AB['T_Purchase'], AB['C_Purchase'])

# LeveneResult(statistic=2.6392694728747363, pvalue=0.10828588271874791)
# h0 isn't rejected.

######################################################
# Hypothesis Testing

test_statistics, pvalue = stats.ttest_ind(AB['T_Purchase'], AB['C_Purchase'], equal_var=True)
print('test statistics = %.4f, p-value = %.4f' % (test_statistics, pvalue))

#test statistics = 0.9416, p-value = 0.3493
# h0 isn't rejected.



#######################################
# # Create hypothesis (Earning)
#######################################

# Control and Test Purchase mean:
# C_Earning:    1908.5683
# T_Earning:   2514.890733

# H0: M1 = M2 There is no statistically significant difference between the maximum bidding and average bidding.
# H1: M1 != M2 There is a statistically significant difference between the maximum bidding and average bidding.


# Looking some graphic

A_E = control[['Earning']]
B_E = test[['Earning']]
AB_E = pd.concat([A_E, B_E], axis=1)
AB_E.columns = ['C_Earning', 'T_Earning']
AB_E.head()
AB_E.mean()
sns.kdeplot(data=AB_E, shade=True)
plt.show()


group_ae = pd.DataFrame(np.arange(len(A_E)))
group_ae[:] = 'AE'
AEG = pd.concat([A_E, group_ae], axis=1)
group_be = pd.DataFrame(np.arange(len(B_E)))
group_be[:] = 'BE'
BEG = pd.concat([B_E, group_be], axis=1)
AEB = pd.concat([AEG, BEG])
AEB.columns = ['Earning', 'Groups']
sns.boxplot(x='Groups', y='Earning', data=AEB)
plt.show()

# Assumptions of normality
# H0 = Normal distribution assumption is provided.
# H1 = Normal distribution assumption is not provided.

test_statistics, pvalue = shapiro(AB_E['C_Earning'])
print('test statistics = %.4f, p-value = %.4f' % (test_statistics, pvalue))

# h0 isn't rejected.


test_statistics, pvalue = shapiro(AB_E['T_Earning'])
print('test statistics = %.4f, p-value = %.4f' % (test_statistics, pvalue))

# h0 isn't rejected.

###############################################
# Variance homogeneity
# H0 = Variance is homogeneous
# H1 = Variance is not homogeneous.

stats.levene(AB_E['T_Earning'], AB_E['C_Earning'])

# H0 isn't rejected

######################################################
# # Hypothesis Testing

test_statistics, pvalue = stats.ttest_ind(AB_E['T_Earning'], AB_E['C_Earning'], equal_var=True)
print('test statistics = %.4f, p-value = %.4f' % (test_statistics, pvalue))

# H0 rejected






