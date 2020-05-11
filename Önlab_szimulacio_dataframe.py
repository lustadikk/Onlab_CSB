
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import random

# ## ALAP PARAMÉTEREK

user_count = 100
news_count = 1000
starting_credit = 10
random.seed(42)

# ## FELHASZNÁLÓI TULAJDONSÁGOK

def category_to_number_knowledge(row):
    if row['Knowledge'] == 'LOW':
        return np.random.uniform(0,0.45,10) #0,0.45
    elif row['Knowledge'] == 'MID':
        return np.random.uniform(0.3,0.7,10) #0.45,0.65
    elif row['Knowledge'] == 'HIGH':
        return np.random.uniform(0.55,1,10) #0.65,1
    else:
        pass

def category_to_number_risk(risk):
    if risk == 'LOW':
        return random.randint(5,35)
    elif risk == 'MID':
        return random.randint(35,65)
    elif risk == 'HIGH':
        return random.randint(65,95)
    else:
        pass

def news_status(row):
    return random.choice(['Fake','Genuine'])  

# ## SZIMULÁCIÓ FELÉPÍTÉSE

#Felhasználó és hírlisák felépítése
user_list = []
news_list = []
mandatory_news_list = []
for sorszam in range (1, user_count + 1):
    user_list.append('USER_{}'.format(sorszam))
for sorszam in range (1, news_count + 1):
    news_list.append('NEWS_{}'.format(sorszam))
for sorszam in range (1, 21):
    mandatory_news_list.append('MANDATORY_NEWS_{}'.format(sorszam))


#Tulajdonságok meghatározása
credit_list = [starting_credit]*len(user_list)
knowledge_list = []
low_risk = []
mid_risk = []
high_risk = []
risk_list = []
frequency_list = []
count = int(math.floor(user_count/27))
for x in range(int(math.floor(count*27)/3)):
    knowledge_list.extend(['LOW', 'MID', 'HIGH'])
    low_risk.append('LOW')
    mid_risk.append('MID')
    high_risk.append('HIGH')
for _ in range (user_count):
    freq = random.choice(['MEGFONTOLT','ATLAG','LEGNAGYOBB','GYAKRAN','RITKAN','KÖZEPESEN-MEGFONTOLT'])
    frequency_list.append(freq)
risk_list = low_risk + mid_risk + high_risk
if len(knowledge_list) < user_count:
    for _ in range(user_count - len(knowledge_list)):
        knowledge_list.append('MID')
        risk_list.append('MID')


#Dataframe felépítése
df = pd.DataFrame(index=['Status','Status_voted','Cycle','Vector'] + user_list, columns  = ['PI','Multiplier', 'Credit','Knowledge','Risk','Frequency','Actual_vote','Actual_credit','Actual_kozelseg'] + mandatory_news_list + news_list)
df['Credit'][len(df)-user_count:user_count+len(df)-user_count] = credit_list
df['Knowledge'][len(df)-user_count:user_count+len(df)-user_count] = knowledge_list
df['Risk'][len(df)-user_count:user_count+len(df)-user_count] = risk_list
df['Frequency'][len(df)-user_count:user_count+len(df)-user_count] = frequency_list
df['Knowledge'] = df.apply(category_to_number_knowledge,axis = 1)
df.loc['Status'] = df.loc['Status'][len(df.columns)-(news_count + len(mandatory_news_list)):].apply(news_status)
df['Multiplier'][len(df)-user_count:user_count+len(df)-user_count] = 1
df.loc['Vector'] = df.loc['Vector'][len(df.columns)-(news_count + len(mandatory_news_list)):].apply(lambda x: np.random.uniform(0,1,10))


#Hírek közelségének számítása
def kozelseg_szamolas(vector_i,vector_c):
    d = np.dot(vector_i,vector_c)
    v = np.linalg.norm(vector_i)
    w = np.linalg.norm(vector_c)
    a = d/(v*w)
    return a
df_kozelseg = pd.DataFrame(index = ['Vector_I'] + (mandatory_news_list + news_list), columns = ['Vector_C']+(mandatory_news_list + news_list))
df_kozelseg.loc['Vector_I'] = df.loc['Vector'][9:]
df_kozelseg['Vector_C'] = df_kozelseg.loc['Vector_I'].transpose()
col_list = df_kozelseg.columns.tolist()[1:]
col_list
for col in col_list:
    df_kozelseg[col][1:]= df_kozelseg[1:].apply(lambda x: kozelseg_szamolas(df_kozelseg.at['Vector_I',col],x['Vector_C']),axis = 1)


# Előzetes hírértékelés
def voting_mandatory(knowledge,vector,vote):
    array_difference = knowledge - vector
    boolean_array_list = array_difference.tolist()
    vote_list = []
    if vote == 'Fake':
        for elem in boolean_array_list:
            if elem <= 0:
                #ha nem tudja a felhasználó a választ, akkor véletlenszerű válasz lehetőséget generál
                vote_list.append(random.choice(['Genuine','Fake']))
            else:
                vote_list.append('Fake')
    else:
        for elem in boolean_array_list:
            if elem <= 0:
                #ha nem tudja a felhasználó a választ, akkor véletlenszerű válasz lehetőséget generál
                vote_list.append(random.choice(['Genuine','Fake']))        
            else:
                vote_list.append('Genuine')
    return vote_list

def multiplier_change(vote, col_vote):
    #Ha eltalálja az előzetes hírértékelés szavazást, akkor 1-gyel nő a szorzója
    if col_vote == vote:
        return 1
    else:
        return 0

mandatory_cols = [col for col in df if col.startswith('MANDATORY_NEWS')]
for col in mandatory_cols:
    vote = df.loc['Status'][df.columns.get_loc(col)]
    df[col][len(df)-user_count:] = df[len(df)-user_count:].apply(lambda x: random.choice(voting_mandatory(x['Knowledge'], df.at['Vector',col], vote)),axis = 1)
    df['Multiplier'][len(df)-user_count:] += df[len(df)-user_count:].apply(lambda x: multiplier_change(vote,x[col]),axis = 1)


# Felhasználók tulajdonságai
def status_voting(status,news_vector,knowledge_vector):
    if status == 'MEGFONTOLT':
        if (knowledge_vector > news_vector).all():
            return 1
        else:
            return 0
    elif status == 'KÖZEPESEN-MEGFONTOLT':
        summa = 0
        for i in range(len(knowledge_vector)):
            if knowledge_vector[i] > news_vector[i]:
                summa += 1
            else:
                pass
        if summa > 5:
            return 1
        else:
            return 0
    elif status == 'LEGNAGYOBB':
        if news_vector.max() < knowledge_vector[news_vector.argmax()]:
            return 1
        else:
            return 0
    elif status == 'ATLAG':
        if knowledge_vector.mean() > news_vector.mean():
            return 1
        else:
            return 0
    elif status == 'LEGKISEBB':
        if knowledge_vector.min() > news_vector[knowledge_vector.argmin()]:
            return 1
        else:
            return 0
    elif status == 'GYAKRAN':
        if random.randint(0,100) < 50:
            return 1
        else:
            return 0
    elif status == 'RITKAN':
        if random.randint(0,100) < 25:
            return 1
        else:
            return 0

# ## HÍREK KIÉRTÉKELÉSE FÜGGVÉNY

def voting_status(df,user_list,col,rate):
    vote = df.loc['Status'][df.columns.get_loc(col)]
    df['Actual_vote'] = df.apply(lambda x: status_voting(x['Frequency'],df.at['Vector',col],x['Knowledge']),axis = 1)
    for user in user_list:
        if df.at[user,'Actual_vote'] == 1 and df.at[user,'Credit'] > 0 :
            list_values = df.loc[user][9:].dropna().tolist()
            list_indexes = df.loc[user][9:].dropna().index.tolist()
            summ_kozelseg = 0
            for i in range(len(list_values)):
                kozelseg_ertek = df_kozelseg.at[list_indexes[i], col]
                if list_values[i] == df.at['Status',col]:
                    summ_kozelseg += kozelseg_ertek
                else:
                    pass
            df.at[user,'Actual_kozelseg'] = summ_kozelseg
            df.at[user,col] = random.choice(voting_mandatory(df.at[user,'Knowledge'], df.at['Vector',col],vote))
            df.at[user,'Actual_credit'] = 0
            df.at[user,'Actual_credit'] = int(df.at[user,'Credit']) * (category_to_number_risk(df.at[user,'Risk'])/100)
        else:
            pass
    count_fake = 0
    count_genuine =0
    credit_fake = 0
    credit_genuine = 0
    arany_szorzo_fake = 0
    arany_szorzo_genuine = 0
    list_answers = df[col][4:].dropna().tolist()
    list_users_who_answered =df[col][4:].dropna().index.tolist()
    for i in range(len(list_answers)):
        if list_answers[i] == 'Fake':
            # a count_fake adja meg a FAKE állapotra szavazók súlyát
            count_fake += (df.at[list_users_who_answered[i],'Actual_kozelseg']+df.at[list_users_who_answered[i],'Multiplier']*df.at[list_users_who_answered[i],'Credit'])
            credit_fake += df.at[list_users_who_answered[i],'Actual_credit']
            arany_szorzo_fake += (df.at[list_users_who_answered[i],'Actual_credit'] / df.at[list_users_who_answered[i],'Credit'])
        else:
            count_genuine += (df.at[list_users_who_answered[i],'Actual_kozelseg']+df.at[list_users_who_answered[i],'Multiplier']*df.at[list_users_who_answered[i],'Credit'])
            credit_genuine += (df.at[list_users_who_answered[i],'Actual_credit'])
            arany_szorzo_genuine += (df.at[list_users_who_answered[i],'Actual_credit'] / df.at[list_users_who_answered[i],'Credit'])
    
    if count_fake > count_genuine and count_fake/(count_fake + count_genuine) > rate:
        df.at['Cycle',col] = 'DONE'
        df.at['Status_voted',col] = 'Fake'
        for i in range(len(list_answers)):
            if list_answers[i] == 'Fake':
                df.at[list_users_who_answered[i],'Credit'] += ((credit_genuine * (count_fake/(count_fake + count_genuine)))/arany_szorzo_fake)*(df.at[list_users_who_answered[i],'Actual_credit'] / df.at[list_users_who_answered[i],'Credit'])
                if df.at[list_users_who_answered[i],'Multiplier'] < 10:
                    df.at[list_users_who_answered[i],'Multiplier'] += (count_fake/(count_fake + count_genuine)) - (count_genuine/(count_fake + count_genuine))
                else:
                    df.at[list_users_who_answered[i],'Multiplier'] += ((count_fake/(count_fake + count_genuine)) - (count_genuine/(count_fake + count_genuine)))/100
            else:
                if arany_szorzo_genuine != 0:
                    df.at[list_users_who_answered[i],'Credit'] += ((credit_genuine * (count_genuine/(count_fake + count_genuine)))/arany_szorzo_genuine)*(df.at[list_users_who_answered[i],'Actual_credit'] / df.at[list_users_who_answered[i],'Credit']) - df.at[list_users_who_answered[i],'Actual_credit']
                    #df.at[list_users_who_answered[i],'Multiplier'] -= ((count_fake/(count_fake + count_genuine)) - (count_genuine/(count_fake + count_genuine)))*3
                    df.at[list_users_who_answered[i],'Multiplier'] = (df.at[list_users_who_answered[i],'Multiplier'])/3
                    #df.at[list_users_who_answered[i],'Multiplier'] = (2*df.at[list_users_who_answered[i],'Multiplier'])/3
                else:
                    pass
                # if df.at[list_users_who_answered[i],'Multiplier'] < 1:
                #     df.at[list_users_who_answered[i],'Multiplier'] = 1

    elif count_genuine > count_fake and count_genuine/(count_fake + count_genuine) > rate:
        df.at['Cycle',col] = (1-rate)*10
        df.at['Status_voted',col] = 'Genuine'
        for i in range(len(list_answers)):
            if list_answers[i] == 'Genuine':
                df.at[list_users_who_answered[i],'Credit'] += ((credit_fake * (count_genuine/(count_fake + count_genuine)))/arany_szorzo_genuine)*((df.at[list_users_who_answered[i],'Actual_credit']) / (df.at[list_users_who_answered[i],'Credit']))
                if df.at[list_users_who_answered[i],'Multiplier'] < 10:
                    df.at[list_users_who_answered[i],'Multiplier'] += (count_genuine/(count_fake + count_genuine)) - (count_fake/(count_fake + count_genuine))
                else:
                    df.at[list_users_who_answered[i],'Multiplier'] += ((count_genuine/(count_fake + count_genuine)) - (count_fake/(count_fake + count_genuine)))/100
            else:
                if arany_szorzo_fake != 0:
                    df.at[list_users_who_answered[i],'Credit'] += ((credit_fake * (count_fake/(count_fake + count_genuine)))/arany_szorzo_fake)*((df.at[list_users_who_answered[i],'Actual_credit']) / (df.at[list_users_who_answered[i],'Credit'])) - df.at[list_users_who_answered[i],'Actual_credit']
                    #df.at[list_users_who_answered[i],'Multiplier'] -= ((count_genuine/(count_fake + count_genuine)) - (count_fake/(count_fake + count_genuine)))*3
                    #df.at[list_users_who_answered[i],'Multiplier'] = (2*df.at[list_users_who_answered[i],'Multiplier'])/3
                    df.at[list_users_who_answered[i],'Multiplier'] = (df.at[list_users_who_answered[i],'Multiplier'])/3
                else:
                    pass
                # if df.at[list_users_who_answered[i],'Multiplier'] < 1:
                #     df.at[list_users_who_answered[i],'Multiplier'] = 1
    else:
        df.at['Status_voted',col] = 'NO DECISION'
        return
        


news_cols = [col for col in df if col.startswith('NEWS')]
rate = 0.8
for col in news_cols:
    voting_status(df,user_list,col,rate)


count_voted = 0
for col in news_cols:
    if df.at['Status',col] == df.at['Status_voted',col]:
        count_voted +=1
    else:
        pass
print('A sikeresen megszavazott hírek aránya: {}%'.format(count_voted/len(news_list)*100))


print(df.loc['Status_voted'].value_counts())


print(df.sort_values(by = ['Credit'], ascending=False).head(50))

