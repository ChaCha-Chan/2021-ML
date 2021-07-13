import pandas as pd
import re
import nltk
from nltk.corpus import stopwords



mbti_dic = {'E': 0, 'I': 1, 'S': 0, 'N': 1, 'T': 0, 'F': 1, 'J': 0, 'P': 1}


def posts_process(posts):
    post_string = (' '.join(posts.strip('\'').split('|||'))).lower()
    post_list = post_string.split()
    for i in range(len(post_list)):
        post_list[i] = re.sub('http(.*)', '', post_list[i])
        post_list[i] = re.sub('(e|i)(s|n)(t|f)(j|p)', '', post_list[i])
        post_list[i] = re.sub('[^a-z]', '', post_list[i])

    stop_words = set(stopwords.words('english'))
    post_list = [word for word in post_list if word !=
                 '' and word not in stop_words and len(word) > 2]
    post_string = ' '.join(post_list)
    return post_string


def read_data():
    data = pd.read_csv('mbti_1.csv')
    data['EI'] = data['type'].apply(lambda x: mbti_dic[x[0]])
    data['SN'] = data['type'].apply(lambda x: mbti_dic[x[1]])
    data['TF'] = data['type'].apply(lambda x: mbti_dic[x[2]])
    data['JP'] = data['type'].apply(lambda x: mbti_dic[x[3]])
    data['posts'] = data['posts'].apply(posts_process)
    for i in range(len(data)-1, -1, -1):
        if data['posts'][i] == '':
            print(i, 'is nan')
            data.drop(index=i, inplace=True)
    data.reset_index(drop=True, inplace=True)   
    data.to_csv('processed.csv')

if __name__ == '__main__':
    read_data()
