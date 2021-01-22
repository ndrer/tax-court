from nltk.tokenize import word_tokenize 
from nltk.probability import FreqDist
import pandas as pd

#Prep
df = pd.read_csv(r'D:\Blag - DATA\tax_court_stem.csv',header=0, converters={'no_putusan':str,'jenis_pajak':str,'sengketa':str,'djp_arg':str,'wp_arg':str,'pdpt_majelis':str})
df = df.apply(lambda x: x.astype(str).str.lower())

#tokenizing
def word_tokenize_wrapper(text):
    return word_tokenize(text)

df['djp_arg2'] = df['djp_arg'].apply(word_tokenize_wrapper)

print(df['djp_arg2'].head())

def freqdist_wrapper(text):
    return FreqDist(text)

df['djp_arg3'] = df['djp_arg2'].apply(freqdist_wrapper)
print(df['djp_arg3'].head().apply(lambda x: x.most_common()))