from textnets import Corpus, Textnet
import pandas as pd

#Use python from command not conda

#DJP Win
#Prep
df = pd.read_csv(r'D:\Blag - DATA\taxcourt_djp_win.csv',sep=';',header=0, converters={'no_putusan':str,'jenis_pajak':str,'sengketa':str,'djp_arg':str,'wp_arg':str,'pdpt_majelis':str})
df = df.apply(lambda x: x.astype(str).str.lower())

#DJP argument
corpus_djp = Corpus.from_df(df, doc_col='djp_arg')
tn_djp = Textnet(corpus_djp.tokenized(remove_numbers=False, stem=False, remove=['wpj']), min_docs=10)
words_djp = tn_djp.project(node_type='term')
words_djp.plot(label_nodes=True, color_clusters=True, alpha=0.5, scale_nodes_by='betweenness', node_label_filter=lambda n: n.betweenness() > words_djp.betweenness.median())
tn_djp.plot(label_term_nodes=True, label_doc_nodes=True, show_clusters=True)

#WP argument
corpus_wp = Corpus.from_df(df, doc_col='wp_arg')
tn_wp = Textnet(corpus_wp.tokenized(remove_numbers=False, stem=False, remove=['wpj']), min_docs=10)
words_wp = tn_wp.project(node_type='term')
words_wp.plot(label_nodes=True, color_clusters=True, alpha=0.5, scale_nodes_by='betweenness', node_label_filter=lambda n: n.betweenness() > words_wp.betweenness.median())
tn_wp.plot(label_term_nodes=True, label_doc_nodes=True, show_clusters=True)

#WP Win
#Prep
df = pd.read_csv(r'D:\Blag - DATA\taxcourt_wp_win.csv',sep=';',header=0, converters={'no_putusan':str,'jenis_pajak':str,'sengketa':str,'djp_arg':str,'wp_arg':str,'pdpt_majelis':str})
df = df.apply(lambda x: x.astype(str).str.lower())

#DJP argument
corpus_djp = Corpus.from_df(df, doc_col='djp_arg')
tn_djp = Textnet(corpus_djp.tokenized(remove_numbers=False, stem=False, remove=['wpj']), min_docs=25)
words_djp = tn_djp.project(node_type='term')
words_djp.plot(label_nodes=True, color_clusters=True, alpha=0.5, scale_nodes_by='betweenness', node_label_filter=lambda n: n.betweenness() > words_djp.betweenness.median())
tn_djp.plot(label_term_nodes=True, label_doc_nodes=True, show_clusters=True)

#WP argument
corpus_wp = Corpus.from_df(df, doc_col='wp_arg')
tn_wp = Textnet(corpus_wp.tokenized(remove_numbers=False, stem=False, remove=['wpj']), min_docs=25)
words_wp = tn_wp.project(node_type='term')
words_wp.plot(label_nodes=True, color_clusters=True, alpha=0.5, scale_nodes_by='betweenness', node_label_filter=lambda n: n.betweenness() > words_wp.betweenness.median())
tn_wp.plot(label_term_nodes=True, label_doc_nodes=True, show_clusters=True)

#Partial
#Prep
df = pd.read_csv(r'D:\Blag - DATA\taxcourt_partial.csv',sep=';',header=0, converters={'no_putusan':str,'jenis_pajak':str,'sengketa':str,'djp_arg':str,'wp_arg':str,'pdpt_majelis':str})
df = df.apply(lambda x: x.astype(str).str.lower())

#DJP argument
corpus_djp = Corpus.from_df(df, doc_col='djp_arg')
tn_djp = Textnet(corpus_djp.tokenized(remove_numbers=False, stem=False, remove=['wpj']), min_docs=12)
words_djp = tn_djp.project(node_type='term')
words_djp.plot(label_nodes=True, color_clusters=True, alpha=0.5, scale_nodes_by='betweenness', node_label_filter=lambda n: n.betweenness() > words_djp.betweenness.median())
tn_djp.plot(label_term_nodes=True, label_doc_nodes=True, show_clusters=True)

#WP argument
corpus_wp = Corpus.from_df(df, doc_col='wp_arg')
tn_wp = Textnet(corpus_wp.tokenized(remove_numbers=False, stem=False, remove=['wpj']), min_docs=12)
words_wp = tn_wp.project(node_type='term')
words_wp.plot(label_nodes=True, color_clusters=True, alpha=0.5, scale_nodes_by='betweenness', node_label_filter=lambda n: n.betweenness() > words_wp.betweenness.median())
tn_wp.plot(label_term_nodes=True, label_doc_nodes=True, show_clusters=True)

#trial from CSV to show no putusan
corpus_wp = Corpus.from_csv(r'D:\Blag - DATA\taxcourt_partial.csv',sep=';',label_col='no_putusan', doc_col='wp_arg')