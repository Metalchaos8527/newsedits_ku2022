import os
from subprocess import call
import pandas as pd 
import numpy as np
import sqlite3
from itertools import chain
from tqdm import tqdm

#call db, doc_stats & sents files
def call_db(data_dir, db_file):
    conn = sqlite3.connect(data_dir + db_file)
    return conn

def create_doc_level_stats(conn):

    pd.read_sql('''SELECT 
                    name
                FROM 
                    sqlite_master 
                WHERE 
                    type ='table' AND 
                    name NOT LIKE 'sqlite_%';''', 
                    con=conn)

    doc_level_stats_df = pd.read_sql('select * from doc_level_stats', con=conn)

    #version filtering
    ver_mask = (doc_level_stats_df.version_y < 20)
    filtered_doc_stats = doc_level_stats_df.loc[ver_mask]

    #sentence filtering
    sent_mask = (filtered_doc_stats.num_sentences_y >= 5) & (filtered_doc_stats.num_sentences_y <= 15)
    filtered_doc_stats = filtered_doc_stats.loc[sent_mask]
    filtered_doc_stats = filtered_doc_stats.reset_index(drop = True)

    return doc_level_stats_df, filtered_doc_stats

def create_sents_files(conn):

    sentences = pd.read_sql('select * from split_sentences', con=conn)

    #sort by etnry_id & version in ascending
    sorted_sentences = sentences.sort_values(['entry_id', 'version'], 
                                            ascending = [True, True])
    sorted_sentences = sorted_sentences.reset_index(drop=True)

    return sorted_sentences


#preprocess doc_stats
def get_filtered_eid_versions(filtered_doc_stats):

    filt_entry_ids = filtered_doc_stats['entry_id'].tolist()
    filt_versions = filtered_doc_stats.loc[:, 'version_x':'version_y'].values.tolist()

    uniq_ids, uniq_indexs = np.unique(filt_entry_ids, return_index=True)

    #pariring versions
    paired_indexs = [filt_versions[uniq_indexs[i]:uniq_indexs[i+1]] if i+1 != len(uniq_indexs) else filt_versions[uniq_indexs[i]:] for i in range(len(uniq_indexs))]
    unlist_indexs = [list(set(list(chain(*x)))) for x in paired_indexs ]
    final_indexs = [list(set(x + [min(x)-1])) if min(x) > 0 else x for x in unlist_indexs]

    y_label_indexs = []

    for values in final_indexs:
        
        diffs = []
        new_vals = []
        for i in range(len(values)):
            
            if not i+1 == len(values):
                
                diffs.append(values[i+1] - values[i])
        
        diff_indexs = np.where(np.array(diffs) > 1)[0].tolist()
        
        if len(diff_indexs) > 1:
            
            for index in diff_indexs:
                new_vals.append(values[index+1] -1)
                
            y_label_indexs.append(list(set(values+new_vals)))
            
        else:
            y_label_indexs.append(values)

    id_ver_dict = {id_:version for id_,version in zip(uniq_ids, y_label_indexs)}

    return id_ver_dict


#preprocess sents
def preprocess_sentences(sorted_sentences, id_ver_dict):

    pds = []

    for entry_id, versions in tqdm(id_ver_dict.items()):
        
        floated_versions = [float(x) for x in versions]
        #먼저 entry_id만 추출
        df = sorted_sentences.loc[sorted_sentences.entry_id == entry_id]
        df_vers = df.loc[df.version.isin(floated_versions)]
            
        #versions마다 iter 하면서 sentence 합치기
        uniq_vers = df_vers.version.unique().tolist()
        
        for ver in uniq_vers:
            
            uniq_ver_df = df_vers.loc[df_vers.version == ver]
            
            uniq_ver_sents = "".join(uniq_ver_df.sentence.values.tolist())
            
            raw_val = [[entry_id, ver, uniq_ver_sents]]
            
            df = pd.DataFrame(raw_val, columns = ['entity_id', 'version', 'sentence'])
            pds.append(df)

    final_df = pd.concat(pds)
    final_df = final_df.reset_index(drop=True)

    return final_df

def label_each_cases(doc_level_stats_df, final_df):

    id_queries = final_df.entity_id.values.tolist()

    ver_queries = final_df.version.values.tolist()

    labels = []

    i = 0
    for id_query, ver_query in tqdm(zip(id_queries, ver_queries), total = len(id_queries)):
        
        if ver_query == 0:
            labels.append(0)
            continue
        
        id_mask = (doc_level_stats_df.entry_id == id_query)

        ver_mask = (doc_level_stats_df.version_y == int(ver_query))
        
        try:
            num_additions = doc_level_stats_df.loc[id_mask & ver_mask]['num_added'].values.item()

            if num_additions >= 0 and num_additions < 1:
                label = 0
                
            elif num_additions >= 1 and num_additions < 3:
                label = 1
                
            else:
                label = 2
                
            labels.append(label)
        
        except:
            label = np.nan
            labels.append(label)

    final_df['label'] = labels
    final_df = final_df.dropna()

    return final_df

if __name__=="__main__":

    data_dir = "/data/chaos8527/server14/newsedits_data/matched-sentences/"

    data = "dailymail-matched-sentences.db"

    #connect to db
    conn = call_db(data_dir, data)
    
    #create base doc_stats, filtered_doc_stats and sents
    doc_level_stats_df, filtered_doc_stats = create_doc_level_stats(conn)
    sorted_sentences = create_sents_files(conn)

    id_ver_dict = get_filtered_eid_versions(filtered_doc_stats)

    raw_df = preprocess_sentences(sorted_sentences, id_ver_dict)

    final_df = label_each_cases(doc_level_stats_df, raw_df)

    print(final_df.shape)

