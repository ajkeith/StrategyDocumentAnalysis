import os
import pandas as pd
import pycountry as pc

# load study countries
dir_path = os.path.join(os.getcwd(), 'data', 'metrics')
df = pd.read_csv(os.path.join(dir_path, 'country_study.csv'))
df['alpha3'] = [pc.countries.search_fuzzy(country)[0].alpha_3 for country in df.country]
df.set_index('alpha3', inplace=True)

def load_metric(fn):
    """
    Load metric from csv file.
    """
    df = pd.read_csv(os.path.join(dir_path, fn))
    df['alpha3'] = [pc.countries.search_fuzzy(country)[0].alpha_3 for country in df.country]
    df.drop(columns=['country'], inplace=True)
    if df.alpha3.isna().sum() == 0:
        df.set_index('alpha3', inplace=True)
        return df
    else:
        print('Missing alpha3 code for: ', df[df.alpha3.isna()].country.unique())
        return df

# load metrics
aidv = load_metric('aidv.csv') # higher is better
aigs = load_metric('aigs.csv') # higher is better
cri = load_metric('cri.csv') # higher is better
di = load_metric('di.csv') # higher is better
gai = load_metric('gai.csv') # higher is better
gair = load_metric('gair.csv') # higher is better
gfs = load_metric('gfs.csv') # higher is better
libdem = load_metric('libdem.csv') # higher is better
odi = load_metric('odi.csv') # higher is better
ttaip = load_metric('ttaip.csv') # higher is better

# join metrics
df_summary = df.join(aidv, how='left', rsuffix='_aidv').\
    join(aigs, how='left', rsuffix='_aigs').\
    join(cri, how='left', rsuffix='_cri').\
    join(di, how='left', rsuffix='_di').\
    join(gai, how='left', rsuffix='_gai').\
    join(gair, how='left', rsuffix='_gair').\
    join(gfs, how='left', rsuffix='_gfs').\
    join(libdem, how='left', rsuffix='_libdem').\
    join(odi, how='left', rsuffix='_odi').\
    join(ttaip, how='left', rsuffix='_ttaip').\
    drop_duplicates().\
    groupby(['alpha3','gpai']).mean().reset_index()
df_summary

df_gpai = df_summary[df_summary.gpai == "GPAI"].\
    groupby('gpai').\
    describe().stack().reset_index().\
    rename(columns={'gpai':'alpha3','level_1':'gpai'})

df_out = pd.concat([df_summary[df_summary.gpai != "GPAI"], df_gpai])

# write csv
df_out.to_csv(os.path.join(dir_path, 'metrics.csv'), index=False)
