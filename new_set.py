import pickle
import os

# with open('/home/newuser/Downloads/cornell_movie_dialogs_corpus/dataset_1MM/W.pkl', 'r',encoding='iso-8859-1') as f:
#     data = pickle.load(f)
#
# print(data[:10])


def read_data(path):
    dfs = []
    onlyfiles = [f for f in os.listdir(mypath)]
    for files in onlyfiles:
        fileName = path.join(files)
        print(files)
        internalfiles = [f for f in os.listdir(fileName)]
        for text_files in internalfiles:
            with open(fileName.join(text_files), 'r', encoding='iso-8859-1') as f:
                for line in f:
                    print(line)
                    dfs.append(line)
    return dfs


mypath ='/home/newuser/Downloads/cornell_movie_dialogs_corpus/dialogs/'

data = read_data(mypath)

# df = read_data(flist,
#                date_col='date',
#                sep=r'\s+',
#                header=None,
#                names=['date','prec'],
#                engine='python',
#                skipfooter=1,
#               ) \
#      .replace(replacements, regex=True) \
#      .set_index('date') \
#      .apply(pd.to_numeric, args=('coerce',))

