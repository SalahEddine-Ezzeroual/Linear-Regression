
"""
implementing Linear Regression From Scratch.

Hello world, Please guys, if you Have any questions or extentions 
Please Let me know! Thanks.!!
==> Salah-eddine ezzerouale !

"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


plt.style.use("seaborn")

data_file_path = "data_set\student_scores.csv"

df  = pd.read_csv(data_file_path)

XS = np.array(df.Hours.values, dtype = np.float64)
YS = np.array(df.Scores.values, dtype = np.float64)


def Mean(sequence)  :
  
  return sum(sequence) / len(sequence)


def calc_best_Fit_line(xs, ys)  :

	m_slope = ( ((Mean(xs) * Mean(ys)) - Mean(xs * ys)) /
    ((Mean(xs) ** 2) - Mean(xs **2) ))

	return m_slope 


def Y_intercept(xs,ys,m)  :
  
  return Mean(ys) - m * Mean(xs)


m,b = calc_best_Fit_line(XS,YS), Y_intercept(XS,YS,calc_best_Fit_line(XS,YS))
# print(m,b)


Regression_line = []

for x in XS : 
	Regression_line.append( (m * x) + b )

x_input_predict = 9.5
y_output_prediction = (m * x_input_predict) + b

# print(y_output_prediction) #95.35380561785415

Squared_error = lambda ys_orign, ys_line : sum( (ys_line - ys_orign) ** 2 )

def coefficient_Of_determination(ys_orign, ys_line)  :
	
	y_mean_line = [Mean(ys_orign) for y in ys_orign]
	squared_err_Regression = Squared_error(ys_orign, ys_line)
	squared_err_y_mean = Squared_error(ys_orign,y_mean_line)

	return 1 - (squared_err_Regression / squared_err_y_mean)


R_Squared = coefficient_Of_determination(YS,Regression_line)
print(R_Squared * 100) #95.29481969048355%



plt.scatter(XS,YS, s=100, color="#0D98BA", linewidth=1, alpha=0.75)
plt.scatter(x_input_predict,y_output_prediction, color="#9400D3", s=101)
plt.plot(XS,Regression_line)

plt.title("Linear Regression")
plt.xlabel("Hours")
plt.ylabel("Scores")

plt.show()

