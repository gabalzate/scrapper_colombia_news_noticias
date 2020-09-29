import argparse
import logging
logging.basicConfig(level=logging.INFO)
from urllib.parse import urlparse
import pandas as pd
import hashlib
import nltk
from nltk.corpus import stopwords
logger = logging.getLogger(__name__)
stop_words = set(stopwords.words('spanish'))



def main(filename):
    logger.info('Starting cleaning process')

    df = _read_data(filename)
    newspaper_uid = _extract_newspaper_uid(filename)
    df = _add_newspaper_uid_column(df, newspaper_uid) 
    df = _extract_host(df)
    df = _fill_missing_titles(df)
    df = _generate_uids_for_rows(df)
    df = _remove_new_lines_from_body(df)
    df = _tokenize_title(df)
    df = _tokenize_body(df)
    df = _drop_rows_with_missing_data(df)
    _save_data(df,filename)
    return df


def _read_data(filename):
    logger.info('Reading file {}'.format(filename))
    return pd.read_csv(filename)

def _extract_newspaper_uid(filename):
    logger.info('Extracting newspaper uid')
    newspaper_uid = filename.split('_')[0]
    logger.info('Newspaper uid detected: {}'.format(newspaper_uid))
    return newspaper_uid

def _add_newspaper_uid_column(df, newspaper_uid):
    logger.info('Adding newspaper uid column {}'.format(newspaper_uid))
    df['newspaper_uid'] = newspaper_uid
    return df

def _extract_host(df):
    logger.info('Extracting host from the url newspaper')
    df['host'] = df['articleurl'].apply(lambda articleurl:urlparse(articleurl).netloc)
    return df

def _fill_missing_titles(df):
    logger.info('Filling missing titles')
    missing_titles_mask = df['title'].isna()
    missing_titles = (df[missing_titles_mask]['articleurl']
                        .str.extract(r'(?P<missing_titles>[^/]+)$')
                        .applymap(lambda title: title.split('-'))
                        .applymap(lambda title_word_list: ' '.join(title_word_list))
                        )
    df.loc[missing_titles_mask, 'title'] = missing_titles.loc[:, 'missing_titles']
    return df

def _generate_uids_for_rows(df):
    logger.info('Generating uids for each row')
    uids = (df
            .apply(lambda row: hashlib.md5(bytes(row['articleurl'].encode())), axis = 1)
            .apply(lambda hash_object: hash_object.hexdigest())
            )
    df['uid'] = uids
    return df.set_index('uid')

def _remove_new_lines_from_body(df):
    logger.info('Remove new lines from body')

    stripped_body = (df
                        .apply(lambda row: row['body'], axis = 1)
                        .apply(lambda body: list(body))
                        .apply(lambda letters: list(map(lambda letter: letter.replace('\n', ' '), letters)))
                        .apply(lambda letters: ''.join(letters))
                        )
    df['body'] = stripped_body
    return df

def _tokenize_title(df):
    logger.info('Tokenize title using NL with nltk library')

    tokenize_df_title= (df
                .dropna()
                .apply(lambda row: nltk.word_tokenize(row['title']), axis = 1)
                .apply(lambda tokens: list(filter(lambda token: token.isalpha(), tokens)))
                .apply(lambda tokens: list(map(lambda token: token.lower(), tokens)))
                .apply(lambda word_list: list(filter(lambda word: word not in stop_words, word_list)))
                .apply(lambda valid_word_list: len(valid_word_list))
           )
    df['n_tokens_title'] = tokenize_df_title
    return df

def _tokenize_body(df):
    logger.info('Tokenize body using NL with nltk library')

    tokenize_df_body= (df
                .dropna()
                .apply(lambda row: nltk.word_tokenize(row['body']), axis = 1)
                .apply(lambda tokens: list(filter(lambda token: token.isalpha(), tokens)))
                .apply(lambda tokens: list(map(lambda token: token.lower(), tokens)))
                .apply(lambda word_list: list(filter(lambda word: word not in stop_words, word_list)))
                .apply(lambda valid_word_list: len(valid_word_list))
           )
    df['n_tokens_body'] = tokenize_df_body
    return df

def _drop_rows_with_missing_data(df):
    logger.info('Dropping rows with missing values')
    return df.dropna()

def _save_data(df, filename):
    clean_filename = 'clean_{}'.format(filename)
    logger.info('Saving data at location: {}'.format(clean_filename))
    df.to_csv(clean_filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filename',
                        help = 'The path to the dirty data',
                        type = str)

    args = parser.parse_args()
    df = main(args.filename)
    print(df)
