from textnets import Corpus, Textnet
import pandas as pd

#Use command not conda
#Prep
df = pd.read_csv(r'D:\Blag - DATA\tax_court_stem_2.csv',header=0, converters={'no_putusan':str,'jenis_pajak':str,'sengketa':str,'djp_arg':str,'wp_arg':str,'pdpt_majelis':str})
df = df.apply(lambda x: x.astype(str).str.lower())

#Combined
corpus_all = Corpus.from_df(df, doc_col='argumen')
tn_all = Textnet(corpus_all.tokenized(remove_numbers=False, stem=False, remove=['wpj']), min_docs=50)
words_all = tn_all.project(node_type='term')
words_all.plot(label_nodes=True, color_clusters=True, alpha=0.5, scale_nodes_by='betweenness', node_label_filter=lambda n: n.betweenness() > words_all.betweenness.median())
tn_all.plot(label_term_nodes=True, label_doc_nodes=True, show_clusters=True)

#DJP argument
corpus_djp = Corpus.from_df(df, doc_col='djp_arg')
tn_djp = Textnet(corpus_djp.tokenized(remove_numbers=False, stem=False, remove=['wpj']), min_docs=50)
words_djp = tn_djp.project(node_type='term')
words_djp.plot(label_nodes=True, color_clusters=True, alpha=0.5, scale_nodes_by='betweenness', node_label_filter=lambda n: n.betweenness() > words_all.betweenness.median())
tn_djp.plot(label_term_nodes=True, label_doc_nodes=True, show_clusters=True)

#WP argument
corpus_wp = Corpus.from_df(df, doc_col='wp_arg')
tn_wp = Textnet(corpus_wp.tokenized(remove_numbers=False, stem=False, remove=['wpj']), min_docs=50)
words_wp = tn_wp.project(node_type='term')
words_wp.plot(label_nodes=True, color_clusters=True, alpha=0.5, scale_nodes_by='betweenness', node_label_filter=lambda n: n.betweenness() > words_all.betweenness.median())
tn_wp.plot(label_term_nodes=True, label_doc_nodes=True, show_clusters=True)