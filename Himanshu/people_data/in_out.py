import sys
import pandas as pd
import pickle
import csv
import numpy as np  
import ast


out = open('/Users/prince_13/Documents/projects/try/bakwass/final_/main/out1.txt', 'w')
sys.stdout = out
sys.stderr = out
with open('/Users/prince_13/Documents/projects/try/bakwass/final_/faces_input/common_encodings.pkl', 'rb') as f:
        data = pickle.load(f)
        known_names = [item['name'] for item in data]
    
def data():
    with open('/Users/prince_13/Documents/projects/try/bakwass/final_/people_data/data.csv', 'r') as f:
        reader = csv.reader(f)
        parsed_data = []
        for row in reader:
            parsed_data.append(row)
    df = pd.DataFrame(parsed_data)
    return df
def writing(name,ent,time):
    with open('/Users/prince_13/Documents/projects/try/bakwass/final_/log/log.txt', 'a') as f:
        a = f'{name} {ent} at {time}'
        print(a)
        f.write(a)
        f.write('\n')
        
def changing_status(a):
    with open('/Users/prince_13/Documents/projects/try/bakwass/final_/people_data/in_out_present_status.csv','w') as f:
        writer = csv.writer(f)
        writer.writerow(a)
        
def count(df):

    df_data = df.iloc[:,:-1]
    last_30 = df_data.tail(30)
    count = last_30.apply(pd.Series.value_counts)
    count_values = []
    for i in count.index :
        count_values.append(i)
    
    with open('/Users/prince_13/Documents/projects/try/bakwass/final_/people_data/in_out_present_status.csv','r') as f:
        reader = csv.reader(f)
        a = []
        for row in reader :
            a.append(row)
        a=np.squeeze(np.array(a))
        a = [np.squeeze(ast.literal_eval(x)) for x in a]
        a = [x.item() for x in a]

    count.to_numpy()
    if len(count_values) == 2:
        # print(count)
        for i in range (len(known_names)):
        
            # print(count.iloc[0, i], count.iloc[1, i])
            if count.iloc[0,i] == 30  and a[i] == True:
                writing(known_names[i],'exited',df.iloc[-1,-1])
                a[i] = False
                changing_status(a)
            elif count.iloc[1,i] == 30 and a[i] == False:
                writing(known_names[i],'entered',df.iloc[-1,-1])
                a[i] = True
                changing_status(a)
    else:
        if count_values[0] == True :
            for i in range (len(known_names)):
                if  a[i] == False:
                    writing(known_names[i],'entered',df.iloc[-1,-1])
                    a[i] = True
                    changing_status(a)
        else:
            for i in range (len(known_names)):
                if a[i] == True:
                    writing(known_names[i],'exited',df.iloc[-1,-1])
                    a[i] = False
                    changing_status(a)
            
            
    
            
def record():
    df = data()
    count(df)