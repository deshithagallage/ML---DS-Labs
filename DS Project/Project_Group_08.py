import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer


# Load the data
attendance_df = pd.read_csv('attendance.csv')
employees_df = pd.read_csv('employees.csv')
holidays_df = pd.read_csv('holidays.csv')
leaves_df = pd.read_csv('leaves.csv')
salary_df = pd.read_csv('salary.csv')
salary_dictionary_df = pd.read_csv('salary_dictionary.csv')

# Get copy of the data
Marvellous = employees_df.copy()
employees = employees_df.copy()


# Replace '\N' and '00-00-0000' with NaN
employees.replace({r'\N': np.nan, '00-00-0000': np.nan, "0000-00-00":np.nan, "'0000'": np.nan}, inplace=True)
Marvellous.replace({r'\N': np.nan, '00-00-0000': np.nan, "0000-00-00":np.nan, "'0000'": np.nan}, inplace=True)


unique_titles = [employees['Title'].unique(), employees['Gender'].unique(), employees['Marital_Status'].unique(), employees['Employment_Category'].unique(), employees['Status'].unique(), employees['Employment_Type'].unique(), employees['Religion'].unique() ]


# Add a new column 'Is_resigned' to the data
employees['Is_resigned'] = np.where(employees['Date_Resigned'].isnull(), 0, 1)
Marvellous['Is_resigned'] = np.where(Marvellous['Date_Resigned'].isnull(), 0, 1)


# Convert the 'Date_Joined', 'Date_Resigned' and 'Inactive_Date' columns to datetime data type
Marvellous['Date_Joined'] = pd.to_datetime(Marvellous['Date_Joined'])
Marvellous['Date_Resigned'] = pd.to_datetime(Marvellous['Date_Resigned'])
Marvellous['Inactive_Date'] = pd.to_datetime(Marvellous['Inactive_Date'])

employees['Year_Joined'] = pd.to_datetime(employees['Date_Joined']).dt.year
employees['Month_Joined'] = pd.to_datetime(employees['Date_Joined']).dt.month
employees['Day_Joined'] = pd.to_datetime(employees['Date_Joined']).dt.day


# Imput NaN values in 'Date_Resigned' column with '1/1/2030'
employees['Date_Resigned'] = employees['Date_Resigned'].fillna('1/1/2030')

employees['Year_Resigned'] = pd.to_datetime(employees['Date_Resigned']).dt.year
employees['Month_Resigned'] = pd.to_datetime(employees['Date_Resigned']).dt.month
employees['Day_Resigned'] = pd.to_datetime(employees['Date_Resigned']).dt.day


# Imput NaN values in 'Date_Resigned' column with '1/1/2030'
employees['Inactive_Date'] = employees['Inactive_Date'].fillna('1/1/2030')

employees['Inactive_Year'] = pd.to_datetime(employees['Inactive_Date']).dt.year
employees['Inactive_Month'] = pd.to_datetime(employees['Inactive_Date']).dt.month
employees['Inactive_Day'] = pd.to_datetime(employees['Inactive_Date']).dt.day


# Convert the new columns to integer data type
int_columns = ['Year_Resigned', 'Month_Resigned', 'Day_Resigned', 'Inactive_Year', 'Inactive_Month', 'Inactive_Day']
employees[int_columns] = employees[int_columns].astype('Int64')


employees = employees.drop('Reporting_emp_1', axis=1)
Marvellous = Marvellous.drop('Reporting_emp_1', axis=1)

employees = employees.drop('Reporting_emp_2', axis=1)
Marvellous = Marvellous.drop('Reporting_emp_2', axis=1)


cols_to_drop = ['Name', 'Religion_ID', 'Designation_ID']
Marvellous.drop(columns=cols_to_drop, inplace=True)

cols_to_drop = ['Employee_Code', 'Name', 'Religion_ID', 'Designation_ID', 'Date_Joined', 'Date_Resigned', 'Inactive_Date']
employees.drop(columns=cols_to_drop, inplace=True)


cols_to_OHEncode = ['Title', 'Religion']

# One-hot encode the specified columns with desired column names
employees = pd.get_dummies(employees, columns=cols_to_OHEncode, prefix=cols_to_OHEncode)

# Convert one-hot encoded columns to integers (1s and 0s)
for col in employees.columns:
    if col.startswith(tuple(cols_to_OHEncode)):
        employees[col] = employees[col].astype(int)


binary_cat_cols = ['Gender', 'Marital_Status', 'Status', 'Employment_Type']

# Mapping for binary categorical variables
mapping = {'Male': 1, 'Female': 0, 'Married': 1, 'Single': 0, 'Active': 1, 'Inactive': 0, 'Permanant': 1, 'Contarct Basis': 0}

# Convert binary categorical variables to 1 and 0
for col in binary_cat_cols:
    employees[col] = employees[col].map(mapping).fillna(np.nan)

# Rename columns if possible
new_column_names = {'Gender': 'Is_Male', 'Marital_Status': 'Is_Married', 'Status': 'Is_Active', 'Employment_Type': 'Is_Permanent'}
employees.rename(columns=new_column_names, inplace=True)


mapping_ordinal = {'Management': 3, 'Staff': 2, 'Labour': 1}
employees['Employment_Category'] = employees['Employment_Category'].map(mapping_ordinal)


salary = salary_df.copy()

# Add Employee salary to the dataset (employees)
filtered_salary = salary[(salary['Basic Salary_0'] != 0)]
mean_basic_salaries = filtered_salary.groupby('Employee_No')['Basic Salary_0'].mean().round(2).reset_index()
employees_with_salary = pd.merge(Marvellous, mean_basic_salaries, on='Employee_No', how='left')

filtered_salary = salary[(salary['Net Salary'] != 0)]
mean_net_salaries = filtered_salary.groupby('Employee_No')['Net Salary'].mean().round(2).reset_index()
employees_with_salary = pd.merge(employees_with_salary, mean_net_salaries, on='Employee_No', how='left')

# Rename the column to 'Basic Salary'
employees_with_salary.rename(columns={'Basic Salary_0': 'Basic Salary'}, inplace=True)


attendance = attendance_df.copy()

# Remove outlier values
attendance['date'] = pd.to_datetime(attendance['date'])
attendance_filtered = attendance[attendance['date'] <= '2023-12-31']

# Add Attendance to the dataset (employees)
attendance_unique_values = attendance_filtered['Employee_No'].value_counts().reset_index()
attendance_unique_values.columns = ['Employee_No', 'Attendance']

employees_with_attendance = pd.merge(employees_with_salary, attendance_unique_values, on='Employee_No', how='left')


# Get unique designation categories
designation_categories = employees_with_attendance['Designation'].unique()

def categories_median_imptution(dataframe, col_to_impute, hue, threshold):
    # Calculate the median of Basic Salary for the entire dataset
    global_median_salary = dataframe[col_to_impute].median().round(2)

    # Initialize a dictionary to store category-wise medians
    category_medians = {}

    # Calculate category-wise medians
    for category in designation_categories:
        category_data = dataframe[dataframe[hue] == category]
        non_null_values_count = category_data[col_to_impute].notnull().sum()

        if non_null_values_count > threshold:
            # Calculate median of Basic Salary within the category
            category_median_salary = category_data[col_to_impute].median()
            category_medians[category] = category_median_salary.round(2)

    # Impute missing values based on category-wise or global median
    imputed_salaries = []
    for index, row in dataframe.iterrows():
        category = row[hue]
        basic_salary = row[col_to_impute]

        if pd.isnull(basic_salary):
            if category in category_medians:
                imputed_salaries.append(category_medians[category])
            else:
                imputed_salaries.append(global_median_salary)
        else:
            imputed_salaries.append(basic_salary)

    # Update the col_to_impute column with imputed values
    dataframe[col_to_impute] = imputed_salaries
    return dataframe

# Impute null valuse in 'Basic Salary' and 'Net Salary'
employees_with_attendance = categories_median_imptution(employees_with_attendance, 'Basic Salary', 'Designation', 10)
employees_with_attendance = categories_median_imptution(employees_with_attendance, 'Net Salary', 'Designation', 10)


# Impute missing values with the median and convert to integer
attendance_median = employees_with_attendance['Attendance'].median()
employees_with_attendance['Attendance'] = employees_with_attendance['Attendance'].fillna(attendance_median).astype(int)


Marvellous = employees_with_attendance


# Take a copy of the data and drop the columns 'Is_Married'
employee_yob = employees.copy()
employee_yob.drop('Is_Married', axis=1, inplace=True)

# Take a copy of the data and drop the columns 'Year_of_Birth'
employee_married = employees.copy()
employee_married.drop('Year_of_Birth', axis=1, inplace=True)


# Drop the column 'Designation' from the data
employee_yob.drop('Designation', axis=1, inplace=True)
employee_married.drop('Designation', axis=1, inplace=True)


# Implement a knn imputer to impute missing values in employee_yob data set exclude 'Employee_No' column
imputer = KNNImputer(n_neighbors=5)
employee_yob_imputed = imputer.fit_transform(employee_yob.iloc[:, 1:])
employee_yob_imputed = pd.DataFrame(employee_yob_imputed, columns=employee_yob.iloc[:, 1:].columns)
employee_yob_imputed['Employee_No'] = employee_yob['Employee_No']


# Implement a knn imputer to impute missing values in employee_married data set exclude 'Employee_No' column
imputer = KNNImputer(n_neighbors=33)
employee_married_imputed = imputer.fit_transform(employee_married.iloc[:, 1:])
employee_married_imputed = pd.DataFrame(employee_married_imputed, columns=employee_married.iloc[:, 1:].columns)
employee_married_imputed['Employee_No'] = employee_married['Employee_No']


employee_yob_imputed['Year_of_Birth'] = employee_yob_imputed['Year_of_Birth'].astype(int)
employee_married_imputed['Is_Married'] = employee_married_imputed['Is_Married'].astype(int)


# Map the 'Is_Married' column to 'Married' and 'Single'
employee_married_imputed['Is_Married'] = employee_married_imputed['Is_Married'].map({1: 'Married', 0: 'Single'})

Marvellous['Year_of_Birth'] = employee_yob_imputed['Year_of_Birth']
Marvellous['Marital_Status'] = employee_married_imputed['Is_Married']


# Outputting DataFrame to a CSV file
# Project Group 08
Marvellous.to_csv('employee_preprocess_08.csv', index=False)
