import csv
import pandas as pd
import re
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover, ArrayDictionary
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


#Prep
df = pd.read_excel(r'D:\Blag - DATA\tax_court_clean.xlsx', engine='openpyxl', sheet_name='Sheet1',header=0, converters={'no_putusan':str,'jenis_pajak':str,'sengketa':str,'djp_arg':str,'wp_arg':str,'pdpt_majelis':str})
df = df.apply(lambda x: x.astype(str).str.lower())

def filtering(clean):
    clean = re.sub('[\W_]+', ' ', clean)
    clean = re.sub('\d+', ' ', clean)
    clean = re.sub('\s+', ' ', clean)
    clean = re.sub('\b[a-zA-Z]\b',' ', clean)
    clean = re.sub(' +', ' ',clean)
    return clean

df['sengketa'] = df['sengketa'].apply(filtering)
df['djp_arg'] = df['djp_arg'].apply(filtering)
df['wp_arg'] = df['wp_arg'].apply(filtering)
df['pdpt_majelis'] = df['pdpt_majelis'].apply(filtering)

#Remove stopwords

factory_sw = StopWordRemoverFactory()
stopwords = factory_sw.get_stop_words()

new_stop = []
with open('D:\Blag - DATA\swindo.txt') as inputfile:
    for row in csv.reader(inputfile):
        new_stop.append(row[0])


sw_combine = factory_sw.get_stop_words()+new_stop
dictionary = ArrayDictionary(sw_combine)
stop = StopWordRemover(dictionary)

df['sengketa'] = df['sengketa'].apply(lambda x: stop.remove(x))
df['djp_arg'] = df['djp_arg'].apply(lambda x: stop.remove(x))
df['wp_arg'] = df['wp_arg'].apply(lambda x: stop.remove(x))
df['pdpt_majelis'] = df['pdpt_majelis'].apply(lambda x: stop.remove(x))
df['sengketa'] = df['sengketa'].apply(lambda x: stop.remove(x))
df['djp_arg'] = df['djp_arg'].apply(lambda x: stop.remove(x))
df['wp_arg'] = df['wp_arg'].apply(lambda x: stop.remove(x))
df['pdpt_majelis'] = df['pdpt_majelis'].apply(lambda x: stop.remove(x))

#lemmatizing or stemming
factory_st = StemmerFactory()
stemmer = factory_st.create_stemmer()


df['sengketa'] = df['sengketa'].apply(lambda x: stemmer.stem(x))
df['djp_arg'] = df['djp_arg'].apply(lambda x: stemmer.stem(x))
df['wp_arg'] = df['wp_arg'].apply(lambda x: stemmer.stem(x))
df['pdpt_majelis'] = df['pdpt_majelis'].apply(lambda x: stemmer.stem(x))

df.to_csv(r'D:\Blag - DATA\tax_court_stem.csv', index=False, header=True)