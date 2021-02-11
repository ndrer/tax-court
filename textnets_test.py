from textnets import Corpus, Textnet
import pandas as pd

#Prep
df = pd.read_csv(r'D:\Blag - DATA\tax_court_stem.csv',header=0, converters={'no_putusan':str,'jenis_pajak':str,'sengketa':str,'djp_arg':str,'wp_arg':str,'pdpt_majelis':str})
df = df.apply(lambda x: x.astype(str).str.lower())

#tokenizing

corpus = Corpus.from_df(df, doc_col='djp_arg')
tn = Textnet(corpus.tokenized(), min_docs=1)

tn.plot(label_term_nodes=True, label_doc_nodes=True, show_clusters=True)
words = tn.project(node_type='term')
words.plot(label_nodes=True, show_clusters=True)