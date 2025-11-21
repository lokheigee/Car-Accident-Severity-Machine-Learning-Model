import pandas as pd
import re


accident = pd.read_csv('accident.csv')
person = pd.read_csv('person.csv')
person = person[person['ROAD_USER_TYPE'] == 2] #Filtering person for drivers only
vehicle = pd.read_csv('vehicle.csv')

#relevant vars
accident_vars = ['ACCIDENT_NO', 'ACCIDENT_TIME', 'LIGHT_CONDITION', 'ROAD_GEOMETRY', 'SPEED_ZONE', "ASI"]
person_vars = ['ACCIDENT_NO', 'VEHICLE_ID', 'SEX', 'AGE_GROUP', 'LICENCE_STATE', 'HELMET_BELT_WORN']
vehicle_vars = ['ACCIDENT_NO', 'VEHICLE_BODY_STYLE', 'VEHICLE_ID', 'VEHICLE_YEAR_MANUF', 'VEHICLE_WEIGHT', 'NO_OF_WHEELS', 'NO_OF_CYLINDERS' ,'SEATING_CAPACITY', 'TARE_WEIGHT', 'TOTAL_NO_OCCUPANTS','CARRY_CAPACITY','CUBIC_CAPACITY']

#grouping by var type
just_numerical = ['VEHICLE_WEIGHT', 'NO_OF_WHEELS', 'NO_OF_CYLINDERS', 'SEATING_CAPACITY','TARE_WEIGHT', 'AGE_GROUP', 'TOTAL_NO_OCCUPANTS', 'CARRY_CAPACITY', 'CUBIC_CAPACITY','ACCIDENT_TIME', 'SPEED_ZONE']
just_not_numerical = ['VEHICLE_BODY_STYLE', 'VEHICLE_ID', 'VEHICLE_YEAR_MANUF', 'LIGHT_CONDITION', 'ROAD_GEOMETRY','SEX', 'AGE_GROUP', 'LICENCE_STATE', 'HELMET_BELT_WORN']


def combine_dataframes(person_df, vehicle_df, accident_df):
    '''Combines the person, vehicle, and accident dataframes and adds ASI in as the final column'''
    vehicle_accident = vehicle_df[vehicle_vars].merge(accident_df[accident_vars], on='ACCIDENT_NO', how='left')
    
    combined = vehicle_accident.merge(person_df[person_vars], on=['ACCIDENT_NO', 'VEHICLE_ID'])

    last_column = combined.pop('ASI')
 
    combined.insert(len(combined.columns), 'ASI', last_column)

    return combined

def get_ASI(df, ind):
    '''Returns an accident severity score for a given accident'''
    deaths = df['NO_PERSONS_KILLED'][ind]
    ser_injury = df['NO_PERSONS_INJ_2'][ind]
    other_injury = df['NO_PERSONS_INJ_3'][ind]
    total_injury = df['NO_PERSONS'][ind]
    severity = (deaths + 0.5 * ser_injury + 0.25 * other_injury) / total_injury

    return severity

def time_to_float(time):
    '''Converts time into a single integer value'''
    time = str(time)
    time = re.sub(':', ' ', time).split()
    output = int(time[0]) * 60 * 60 + int(time[1]) * 60 + int(time[2])
    return output

def make_to_int(list_of_vals):
    mapping = {}
    counter = 0
    mapped = []
    
    for item in list_of_vals:
        if pd.isna(item):
            mapped.append(item)
        else:
            if item not in mapping:
                mapping[item] = counter
                counter += 1
            mapped.append(mapping[item])
    
    return mapped

def avg_age(age_group):
    '''Assigns a concrete age to each age group'''
    if age_group == "Unknown":
        return
    if age_group == "70+":
        return 70
    age_group = age_group.split('-')
    return (int(age_group[0]) + int(age_group[1])) / 2

def normalize(column):
    '''Normalizes a dataseries'''
    minimum = min(column)
    maximum = max(column)
    return column.apply(lambda x: (x-minimum) / (maximum-minimum))

# Here I make some changes to the dataframes so that each variable is given a numeric value
accident['ACCIDENT_TIME'] = accident['ACCIDENT_TIME'].apply(time_to_float)
person['HELMET_BELT_WORN'] = make_to_int(person['HELMET_BELT_WORN'])
person["SEX"] = make_to_int(person["SEX"])
person["LICENCE_STATE"] = make_to_int(person["LICENCE_STATE"])
person["AGE_GROUP"] = person["AGE_GROUP"].apply(avg_age)
vehicle["VEHICLE_BODY_STYLE"] = make_to_int(vehicle["VEHICLE_BODY_STYLE"])
accident["ASI"] = [get_ASI(accident, i) for i in accident.index]

# Here the dataframes are merged
all_three = combine_dataframes(person, vehicle, accident)

# Sends the new pre-processed dataframe to a csv file
all_three.to_csv('basic_df.csv')

# Normalizes the numerical values of the dataset
for i in just_numerical:
    all_three[i] = normalize(all_three[i])

# Sends the now normalized dataframe to a csv file
all_three.to_csv('normalized_df.csv')

# Sends some variations of the dataframe to two different csv files
just_numerical.append("ASI")
all_three[just_numerical].to_csv('numerical_normalized_df.csv')
just_not_numerical.append("ASI")
all_three[just_not_numerical].to_csv('categorical_df.csv')