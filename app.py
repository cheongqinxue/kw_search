import streamlit as st
st.set_page_config(layout="wide")

import pandas as pd
from rapidfuzz import fuzz
from os.path import join
import s3fs

FS = s3fs.S3FileSystem(anon=False)
DATA_DIR = 's3://qx-poc-public/entity_kw_search/'

processes = {
    'ratio':fuzz.ratio, 
    'partial_ratio':fuzz.partial_ratio, 
    'token_sort_ratio':fuzz.token_sort_ratio,
    'token_set_ratio':fuzz.token_set_ratio,
    'WRatio':fuzz.WRatio}


class KeywordIndexer:
    def __init__(self, df, analyzers):
        self.df = df
        self.analyzers = analyzers
        self.score_columns = [c+'_score' for c,_ in analyzers]
        self.df.fillna('', inplace=True)
        for c, _ in analyzers:
            self.df[c] = self.df[c].str.lower()

    def greedy_process(self, string):
        string = string.lower()
        for column, proc in self.analyzers:
            self.df[column+'_score'] = self.df[column].apply(lambda x: processes[proc](x,string))
        self.df['_score'] = self.df[self.score_columns].max(1)
        return self.df.sort_values(by='_score', ascending=False).head(10)


@st.cache(persist=True, ttl=3600)
def load(which):
    if FS is None:
        df = pd.read_csv(join(DATA_DIR, which+'.csv'), encoding='utf-8', on_bad_lines='warn')
    else:
        with FS.open(join(DATA_DIR, which+'.csv')) as f:
            df = pd.read_csv(f, encoding='utf-8', on_bad_lines='warn')
    return df



def main():

    which = st.sidebar.selectbox(label = 'Select entity type', options=['organizations','persons'], 
        index=0, key=None, help=None)

    df = load(which)

    kw = KeywordIndexer(
        df[['name','concat_names','truncated_name']], 
        [('concat_names','WRatio'), ('truncated_name','ratio')])

    string = st.sidebar.text_input(label='Search')

    result = kw.greedy_process(string)
    st.markdown('### Search results')

    if len(string) > 1:
        st.dataframe(result[['name','_score']].rename(columns={'name':'Name','_score':'Score'}))
    else:
        st.markdown('Enter a query in the side bar to search for names with similar spelling')

if __name__ == '__main__':
    main()
