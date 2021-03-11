import numpy as np
import pandas as pd
import requests
import json


def get_rankings(amount, printList=False):
    """
    Returns list of top engineering graduate schools from US News rankings in order from highest to lowest ranking.

    params:
    amount (int): amount of top rated schools to add to list
    printList (bool): if true, prints list of rankings
    """
    assert (isinstance(amount, int) and 0 <= amount <= 218)
    assert (isinstance(printList, bool))

    rankings = []
    headers = {
        'authority': 'www.usnews.com',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.150 Safari/537.36',
    }
    iter = 1
    page = 1
    while amount > 0:
        resp = requests.get(
            'https://www.usnews.com/best-graduate-schools/api/search?program=top-engineering-schools&specialty=eng&_page=' + str(
                page), headers=headers)
        page += 1
        for item in json.loads(resp.text)['data']['items']:
            if amount > 0:
                uni = item['name'].split(' (')[0].replace('--', ' ')
                if uni in renamedUnis:
                    uni = renamedUnis[uni]
                if printList:
                    print('{}. {}'.format(iter, uni))
                rankings.append(uni)
                iter += 1
                amount -= 1
            else:
                break

    return rankings


def greConversion(dataset, feature):
    """
    covert old GRE score ot new GRE score
    :parma: dataset
    :type: pd.DataFrame
    :param: feature
    type: str
    """
    assert isinstance(dataset, pd.DataFrame)
    assert isinstance(feature, str)
    gre_score = list(dataset[feature])
    for i in range(len(gre_score)):
        if gre_score[i] > 170:
            try:
                if feature == 'greV':
                    gre_score[i] = score_table['newV'][gre_score[i]]
                elif feature == 'greQ':
                    gre_score[i] = score_table['newQ'][gre_score[i]]
            except:
                continue

    return gre_score


def cgpa_conversion(dataset, cgpa="cgpa", cgpaScale='topperCgpa'):
    """
    convert cgpa wrt topperCgpa
    :parma: dataset
    :type: pd.DataFrame
    :param cgpa column name
    :type: str
    :parma: cgpaScale column name
    :type: str
    :return: cgpa
    :type: list
    """
    assert isinstance(dataset, pd.DataFrame)
    assert isinstance(cgpa, str)
    assert isinstance(cgpaScale, str)
    cgpa = dataset[cgpa].tolist()
    cgpaScale = dataset[cgpaScale].tolist()
    for i in range(len(cgpa)):
        if cgpaScale[i] != 0 and cgpaScale[i] is not np.nan:
            cgpa[i] = cgpa[i] / cgpaScale[i]
            # Ratio should <=1, otherwise set to nan
            if cgpa[i] > 1:
                cgpa[i] = np.nan
            elif cgpa[i] == 0:
                cgpa[i] = np.nan
        else:
            cgpa[i] = np.nan

    return cgpa


if __name__ == '__main__':
    # pd.pandas.set_option('display.max_columns', None)
    # pd.pandas.set_option('display.max_rows', None)

    # raw dataset or unprocessed dataset
    data_path = '../Data/original_data.csv'
    dataset = pd.read_csv(data_path)
    # print("Dataset size:", dataset.shape)
    n_sample = dataset.shape[0]
    # raw dataset
    # print(dataset.head())

    # Numerical Variables
    # check if dataset is balanced
    # balanced 52% vs 48%
    accept = len(dataset[dataset['admit'] == 1])
    # print("positive examples:", accept, round(accept / n_sample, 4))
    reject = len(dataset[dataset['admit'] == 0])
    # print("negative examples:", reject, round(reject / n_sample, 4))
    # no missing value in target variable "admit"
    # print((accept + reject) / n_sample)
    ########################################################################################################
    # drop columns
    dataset = dataset.drop(["userName", "userProfileLink",
                            "major", "specialization",
                            "program", "department",
                            "toeflEssay", "greA",
                            "gmatA", "gmatQ", "gmatV", "ugCollege", ], axis=1)

    zero_gpa = dataset[dataset.cgpa == 0]
    termAndYears = dataset['termAndYear']
    dataset['year'] = [int(term.split(' - ')[1]) if isinstance(term, str) and len(term.split(' - ')) == 2 else np.nan
                       for
                       term in termAndYears]
    dataset = dataset.drop(["termAndYear"], axis=1)

    renamedUnis = {'Virginia Tech': 'Virginia Polytechnic Institute and State University',
                   'University of Wisconsin--Madison': 'University of Wisconsin Madison',
                   'Texas A&M University College Station': 'Texas A and M University College Station',
                   'Stony Brook University SUNY': 'SUNY Stony Brook',
                   'University at Buffalo SUNY': 'SUNY Buffalo',
                   'Rutgers, The State University of New Jersey New Brunswick': 'Rutgers University New Brunswick/Piscataway',
                   'Purdue University West Lafayette': 'Purdue University',
                   'Ohio State University': 'Ohio State University Columbus'
                   }
    rankings = get_rankings(218, False)
    notFoundTargets = []
    for target in dataset['univName']:
        if target not in rankings and target not in notFoundTargets:
            notFoundTargets.append(target)

    assert (len(notFoundTargets) == 0)
    sorter = dict(zip(rankings, range(1, len(rankings) + 1)))
    dataset['targetRank'] = dataset['univName'].map(sorter)

    # GRE score conversion table
    score_table = pd.read_csv('../Data/score.csv')
    score_table.set_index(['old'], inplace=True)
    # print(score_table.head())

    # perform greConversion
    dataset['greV'] = greConversion(dataset, 'greV')
    dataset['greQ'] = greConversion(dataset, 'greQ')

    # sanity check. we are not remove too many samples
    a = dataset[dataset['greV'] > 170]
    b = dataset[dataset['greQ'] > 170]
    # print("num of abnormal score of greV", len(a))
    # print("num of abnormal score of greQ", len(b))

    # Filter out candidates with greV or greQ > 170 after gre_Conversion
    dataset = dataset[(dataset['greV'] <= 170) & (dataset['greV'] >= 130)]
    dataset = dataset[(dataset['greQ'] <= 170) & (dataset['greQ'] >= 130)]
    # print(dataset.shape)
    # print("remain data percentage:", dataset.shape[0] / n_sample)

    # cgpa_conversion
    dataset['cgpaRatio'] = cgpa_conversion(dataset, cgpa="cgpa", cgpaScale="topperCgpa")

    # drop cgpa and topperCgpa using cgpaRatio instead
    dataset = dataset.drop(['cgpa', 'topperCgpa', 'cgpaScale'], axis=1)

    # element in dataset['journalPubs'] has different type
    dataset['journalPubs'] = dataset['journalPubs'].astype(str)
    for i, element, in enumerate(dataset['journalPubs']):
        # print(type(element))
        # print(i)
        if len(element) > 3:
            dataset['journalPubs'].iloc[i] = 0
    # convert to int64
    dataset['journalPubs'] = dataset['journalPubs'].astype('int64')

    dataset['confPubs'] = dataset['confPubs'].astype(str)
    for i, element, in enumerate(dataset['confPubs']):
        if len(str(element)) > 3:
            dataset['confPubs'].iloc[i] = 0
    # convert to int64
    dataset['confPubs'] = dataset['confPubs'].astype('int64')
    dataset = dataset.drop(["toeflScore"], axis=1)
    ################################################################################################################

    # Missing Values in Numerical Features
    # make a list of features which has missing values
    features_with_na = [feature for feature in dataset.columns if
                        dataset[feature].isnull().sum() > 1 and dataset[feature].dtypes != 'O']
    # print the feature name and the percentage of missing values
    # for feature in features_with_na:
    #     print(feature, '\t\t', np.round(dataset[feature].isnull().mean(), 4) * 100, '\t', '% missing values')

    dataset["cgpaRatio"].fillna(dataset["cgpaRatio"].mean(), inplace=True)
    dataset["internExp"].fillna(dataset["internExp"].mean(), inplace=True)
    ########################################################################################################
    # rearrange columns
    dataset = dataset[['cgpaRatio',
                       'greV', 'greQ',
                       'researchExp', 'industryExp', 'internExp',
                       'journalPubs', 'confPubs',
                       'univName',
                       'year', 'targetRank',
                       'admit'
                       ]]
    ########################################################################################################
    ''' sanity check '''
    # cgpaRatio
    error = dataset[dataset.cgpaRatio < 0.1]
    num_unique = len(dataset.cgpaRatio.unique())

    for i in dataset.cgpaRatio.unique():
        assert isinstance(i, float)
        assert 0 < i <= 1

    # greV
    num_unique = len(dataset.greV.unique())
    for i in dataset.greV.unique():
        assert isinstance(i, float)
        assert 130 <= i <= 170
    # print(num_unique)
    # print(sorted(dataset.greV.unique().astype('int16')))

    # greQ
    num_unique = len(dataset.greQ.unique())
    for i in dataset.greQ.unique():
        assert isinstance(i, float)
        assert 130 <= i <= 170
    # print(num_unique)
    # print(sorted(dataset.greQ.unique().astype('int16')))

    # researchExp
    num_unique = len(dataset.researchExp.unique())
    for i in dataset.researchExp.unique():
        assert (isinstance(i, np.int64))
        assert i >= 0
    # print(num_unique)
    # print(sorted(dataset.researchExp.unique().astype('int16')))

    # industryExp
    num_unique = len(dataset.industryExp.unique())
    for i in dataset.industryExp.unique():
        assert (isinstance(i, np.int64))
        assert i >= 0
    # print(num_unique)
    # print(sorted(dataset.industryExp.unique().astype('int16')))

    # internExp
    num_unique = len(dataset.internExp.unique())
    for i in dataset.internExp.unique():
        assert (isinstance(i, np.float64))
        assert i >= 0
    # print(num_unique)
    # print(sorted(dataset.internExp.unique().astype('int16')))

    # journalPubs
    num_unique = len(dataset.journalPubs.unique())
    for i in dataset.journalPubs.unique():
        assert (isinstance(i, np.int64))
        assert i >= 0
    # print(num_unique)
    # print(sorted(dataset.journalPubs.unique().astype('int16')))

    # confPubs
    num_unique = len(dataset.confPubs.unique())
    for i in dataset.confPubs.unique():
        assert (isinstance(i, np.int64))
        assert i >= 0
    # print(num_unique)
    # print(sorted(dataset.confPubs.unique().astype('int16')))

    # targetRank
    num_unique = len(dataset.targetRank.unique())
    for i in dataset.targetRank.unique():
        assert (isinstance(i, np.int64))
        assert i >= 0
    # print(num_unique)
    # print(sorted(dataset.targetRank.unique().astype('int16')))

    # year
    dataset["year"] = dataset['year'][(dataset['year'] > 1990) & (dataset['year'] < 2020)]
    dataset["year"].fillna(dataset["year"].median(), inplace=True)
    num_unique = len(dataset.year.unique())
    # for i in dataset.year.unique():
    #     assert(isinstance(i,np.int64))
    #     assert i>=0
    # print(num_unique)
    # print(dataset.year.unique())

    print("dataset shape:", dataset.shape)
    print("%data_remaining:", dataset.shape[0] / n_sample * 100)
    print(dataset.head())
    print(dataset.isnull().any())
    print(dataset.info())
    print(dataset.describe())
    # Output to .csv
    SAVE_CSV = False
    if SAVE_CSV:
        dataset.to_csv('clean_data.csv', index=False)
