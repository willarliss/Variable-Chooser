# Variable-Chooser
This project aims to minimize risk of multicollinearity in a set of independent variables for regression analysis.

This project tackles the problem of multicollinearity in regression analysis. The purpose is to identify a subset of variables from a large list of independent variables that can be used in regression analysis to minimize the chance of multicollinearity causing a problem. Provided is a class that automates this task.

The class "Variable Chooser" accepts a list of independent variables, a dependent variables, and a minimum value. The algorithm first creates subsets from the list of independent variables. The subsets consist of every combination from the list of independent variables, ranging in length from k (the number of independent variabes passed to the class) and the minimum combination length passed to the class. Once combinations are made, pearson correlation coefficients are calculated for every pair of variables within each subset/combination. Fisher's z Transformation is applied to each of these coefficients, they are then aggregated by either taking their mean or their weighted average (where weights are determined by each coefficients contribution to the sum of all coefficients within the subset). Once the coefficients are aggregated, the single value is converted back to a pearson coefficient. There is an option at this point to apply a penalty according to the length of the subset/combination. The penalty works by subtracting the following from the coefficient: coefficient / k_indep_vars-combo_length+1.

Aggregated coefficients for the correlation between each variable in a subset and the dependent variable are calculated in the same way. The goal is now to find a subset/combination that simultaneously has the lowest coefficient of aggregated correlation for each pair in the subset as well as the highest coefficient of aggregated correlation for each variable in the subset with the dependent variable. This is found by first changing the later maximization into a minimization by simply subtracting each coefficient from 1. A minimum for these two criteria can be found by calculating the euclidian distance of each combination from 0 (the lowest possible coefficient possible for either set of coefficients). 

![](/sample.png)

The problem is demonstrated graphically below. Each point in the plot represents a subset/combination of variables. The points closest to the origin are those which satisfy the criteria described above.

