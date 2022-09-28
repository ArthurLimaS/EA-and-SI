import numpy as np
import pandas as pd
from itertools import combinations
from scipy.stats import t


def student_t(results, reps, alpha=0.05):
    medias = []
    somas_quadradas = []
    variancias = []
    
    for col in results.columns:
        alg_results = np.array(results[col])

        m = np.mean(alg_results)
        sq = np.sum(alg_results ** 2)
        v = np.var(alg_results)

        medias.append(m)
        somas_quadradas.append(sq)
        variancias.append(v)
    
    df = results.copy()
    tmp_df = pd.DataFrame([medias, somas_quadradas, variancias],
                            columns=results.columns,
                            index=['avg', 'sqrd_sum', 'var'])
    df = df.append(tmp_df)


    columns = []
    ts_criticos = [] # valores de t para cada combinação
    p_values = [] # valor encontrado
    
    comb_array = combinations(results.columns, 2)
    for cmb in comb_array:
        columns.append("{} - {}".format(cmb[0], cmb[1]))

        var1 = df.loc['sqrd_sum'][cmb[0]]
        var2 = df.loc['sqrd_sum'][cmb[1]]
        s = (((reps * var1) + (reps * var2)) / (reps + reps - 2)) \
            * ((reps + reps) / (reps * reps))
        s = np.sqrt(s)

        mean1 = df.loc['avg'][cmb[0]]
        mean2 = df.loc['avg'][cmb[1]]
        p_value = np.abs((mean1 - mean2) / s)

        gl = reps + reps - 2
        pr = 1 - (alpha / 2)
        t_critico = t.ppf(gl, pr)

        ts_criticos.append(t_critico)
        p_values.append(p_value)

    analise = pd.DataFrame(data=[ts_criticos, p_values],
                            columns=columns,
                            index=['t', 'analise'])

    return analise