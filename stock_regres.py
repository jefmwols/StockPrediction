import pandas as pd

date_parser = pd.to_datetime
df = pd.read_csv("AAPL.csv", parse_dates=['Date'])
print(df.head())



# Use only one feature
df_X = df['Date']
df_Y = df['Adj Close']


# Split the data into training/testing sets
df_X_train = df_X[:-20]
df_X_test = df_X[-20:]
# Split the targets into training/testing sets
df_y_train = df_Y[:-20]
df_y_test = df_Y[-20:]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(df_X_train, df_y_train)

# Make predictions using the testing set
df_y_pred = regr.predict(df_X_test)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(df_y_test, df_y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(df_y_test, df_y_pred))

# Plot outputs
plt.scatter(df_X_test, df_y_test,  color='black')
plt.plot(df_X_test, df_y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()