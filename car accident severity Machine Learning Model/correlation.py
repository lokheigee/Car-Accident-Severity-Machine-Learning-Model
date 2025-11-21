import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import LabelEncoder

# Load the numerical normalized dataframe
numerical_normalized_df = pd.read_csv('normalized1_df.csv')

# Drop irrelevant columns
irrelevant_cols = ['VEHICLE_WEIGHT', 'CARRY_CAPACITY', 'CUBIC_CAPACITY', 'Unnamed: 0', 'ACCIDENT_NO', 'VEHICLE_ID']
numerical_normalized_df = numerical_normalized_df.drop(columns=irrelevant_cols)

# Fill missing values for numerical data
numerical_normalized_df = numerical_normalized_df.fillna(numerical_normalized_df.mean())

# Select specific columns for Pearson correlation
selected_columns = ['ACCIDENT_TIME', 'NO_OF_CYLINDERS', 'NO_OF_WHEELS', 'SEATING_CAPACITY', 'TARE_WEIGHT', 'TOTAL_NO_OCCUPANTS', 'ASI']
correlation_matrix = numerical_normalized_df[selected_columns].corr(method='pearson')

# Extract Correlation with ASI
asi_correlations = correlation_matrix['ASI'].sort_values(ascending=False)
print("\nCorrelation of selected numerical features with ASI:")
for feature, correlation in asi_correlations.items():
    print(f"{feature}: {correlation:.4f}")

# Calculate the Pearson correlation for composite feature Occupant_Ratio
numerical_normalized_df['Occupant_Ratio'] = numerical_normalized_df['TOTAL_NO_OCCUPANTS'] / (numerical_normalized_df['SEATING_CAPACITY'] + 1)
numerical_normalized_df['Occupant_Ratio'] = numerical_normalized_df['Occupant_Ratio'].fillna(numerical_normalized_df['Occupant_Ratio'].mean())
corr = numerical_normalized_df['Occupant_Ratio'].corr(numerical_normalized_df['ASI'], method='pearson')
print(f'\nOccupant_Ratio Pearson: {corr:.4f}')

# for MI Mutual Information correlation 
categorical_df = pd.read_csv('categorical1_df.csv')

# Drop irrelevant columns from categorical data
categorical_df = categorical_df.drop(columns=['VEHICLE_ID', 'Unnamed: 0'])

# Fill missing values reliably for categorical data
categorical_df = categorical_df.apply(lambda x: x.fillna(x.mode()[0]) if not x.mode().empty else x)

# Binning ASI for classification models
bins_asi = [-0.001, 0.125, 0.250, 1.001]
labels_asi = ['Low', 'Medium', 'High']
categorical_df['ASI_binned_custom'] = pd.cut(categorical_df['ASI'], bins=bins_asi, labels=labels_asi)

# Discretize AGE_GROUP and VEHICLE_YEAR_MANUF
bins_age = [0, 0.3, 0.6, 1]
labels_age = ['Young', 'Adult', 'Senior']
categorical_df['AGE_GROUP_binned'] = pd.cut(categorical_df['AGE_GROUP'], bins=bins_age, labels=labels_age)

categorical_df['VEHICLE_YEAR_MANUF_binned'] = pd.cut(categorical_df['VEHICLE_YEAR_MANUF'], bins=3, labels=['Old', 'Medium', 'New'])

# Composite Features
label_encoder = LabelEncoder()

# Speed zone + road geometry combo
categorical_df['Speed_Road_Combo'] = numerical_normalized_df['SPEED_ZONE'].astype(str) + "_" + categorical_df['ROAD_GEOMETRY'].astype(str)
categorical_df['Speed_Road_Combo_Encoded'] = label_encoder.fit_transform(categorical_df['Speed_Road_Combo'])

# Light condition + road geometry combo
categorical_df['Light_Road_Combo'] = categorical_df['LIGHT_CONDITION'].astype(str) + "_" + categorical_df['ROAD_GEOMETRY'].astype(str)
categorical_df['Light_Road_Combo_Encoded'] = label_encoder.fit_transform(categorical_df['Light_Road_Combo'])

# Aggregation: Light Road Index
categorical_df['Road_Light_Index'] = categorical_df['LIGHT_CONDITION'].astype(int) + categorical_df['ROAD_GEOMETRY'].astype(int)

# Encode categorical features using LabelEncoder
encoded_df = categorical_df.apply(lambda col: label_encoder.fit_transform(col.astype(str)))

# Calculate Mutual Information for categorical features with ASI
mi_scores = mutual_info_regression(encoded_df, encoded_df['ASI'].values.ravel())
print("\nMutual Information between each categorical feature and ASI:")
for feature, score in zip(encoded_df.columns, mi_scores):
    print(f"{feature}: {score:.4f}")

# Calculate Mutual Information for categorical features with ASI_binned_custom
mi_scores_binned = mutual_info_regression(encoded_df, encoded_df['ASI_binned_custom'].values.ravel())
print("\nMutual Information between each categorical feature and ASI_binned_custom:")
for feature, score in zip(encoded_df.columns, mi_scores_binned):
    print(f"{feature}: {score:.4f}")

# Mutual Information for composite features
composite_features = ['Speed_Road_Combo_Encoded', 'Light_Road_Combo_Encoded', 'Road_Light_Index']
mi_composite = mutual_info_regression(categorical_df[composite_features], categorical_df['ASI'])
print("\nMutual Information for Composite Features with ASI:")
for feature, score in zip(composite_features, mi_composite):
    print(f"{feature}: {score:.4f}")

# Mutual Information and Pearson Correlation for Occupant Ratio
mi_occupant = mutual_info_regression(numerical_normalized_df[['Occupant_Ratio']], numerical_normalized_df['ASI'])[0]
print(f'\nOccupant_Ratio MI: {mi_occupant:.4f}')

