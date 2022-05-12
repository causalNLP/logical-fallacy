import pandas as pd

edu = pd.read_csv('../../data/edu_all_new.csv')[['source_article', 'new_label_2']]
kialo = pd.read_csv('../../data/kialo_all.csv')[['claim', 'labels']]
edu.new_label_2 = edu.new_label_2.apply(lambda x: [x])
edu_v2 = pd.concat([edu.rename(columns={'source_article': 'claim', 'new_label_2': 'labels'}), kialo], ignore_index=True)
edu_v2.to_csv('../../data/edu_v2_all.csv')
