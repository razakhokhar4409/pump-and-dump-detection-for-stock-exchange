import quandl
import numpy as nump
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Fetch the Amazon stock data
adjust = quandl.get("WIKI/AMZN")
# Print Head of Data Fetched
print(adjust.head())
# Store the closely adjusted price
adjust = adjust[['Adj. Close']]
# Print Head of Data adjusted
print(adjust.head())
# Total days to be predicted about data
predict_output = 20  # Total 20 days in future
# Shift up n units
adjust['Prediction'] = adjust[['Adj. Close']].shift(-predict_output)
# print tail of shifted data
print(adjust.tail())
# Create numpy array by converting data frame
x_par = nump.array(adjust.drop(['Prediction'], 1))
# Discard last 20 rows for future use
x_par = x_par[:-predict_output]
print(x_par)
# Create numpy array by converting data frame
y_par = nump.array(adjust['Prediction'])
# Get all data values discarding last 20
y_par = y_par[:-predict_output]
print(y_par)
# Create new SVR
svr_use = SVR(kernel='rbf', C=1e3, gamma=0.1)
# The 80 and 20 ratio is being set to train the data
ver_train, ver_test, hor_train, hor_test = train_test_split(x_par, y_par, test_size=0.2)
# Train the SVR
svr_use.fit(ver_train, hor_train)
# The highest probability is of 1
confidence_interval = svr_use.score(ver_test, hor_test)
print("Confidence Interval (SVM) : ", confidence_interval)
# Create Linear Regression model
linear_model = LinearRegression()
# Train the linear regression model
linear_model.fit(ver_train, hor_train)
# The highest probability is of 1
linear_confidence = linear_model.score(ver_test, hor_test)
print("lr confidence: ", linear_confidence)
# Forecast the original data set to 20 days
predict_ver = nump.array(adjust.drop(['Prediction'], 1))[-predict_output:]
print(predict_ver)
# Print the linear model prediction of next 20 days
predict_linear = linear_model.predict(predict_ver)
print(predict_linear)
# Print SVR model prediction of next 20 days
predict_svm = svr_use.predict(predict_ver)
print(predict_svm)
