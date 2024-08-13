import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from tqdm.auto import tqdm
from collections import defaultdict
from datasets import Dataset

def nli_aggregation(
        d_to_iterate,
        contradiction_agg_method=('top_k_mean', 'top_k_mean'),
        entailment_agg_method=('top_k_mean', 'top_k_mean'),
        neutral_agg_method=('bottom_k_mean', 'median'),
        num_sents_to_avg_over_per_sent={
            'contradiction': 3,
            'entailment': 3,
            'neutral': 3
        },
        num_sents_to_avg_over_per_doc={
            'contradiction': 8,
            'entailment': 8,
        },
        verbose=True
):
    def top_k_mean_agg(g, k, top_or_bottom='top'):
        ascending = False if top_or_bottom == 'top' else True
        return g.apply(lambda s: s.sort_values(ascending=ascending).iloc[:k].mean())

    chunk_size = 200_000
    summary_statistics = []
    for i in tqdm(range(len(d_to_iterate) // chunk_size), disable=not verbose):
        d_i = d_to_iterate[chunk_size * i: chunk_size * (i + 1)]
        df = pd.DataFrame(d_i)
        grouped = df.groupby(['article_url', 'press_release_url', 'press_release_idx'])
        aggs = []
        for col, method in [
            ('contradiction', contradiction_agg_method),
            ('entailment', entailment_agg_method),
            ('neutral', neutral_agg_method)
        ]:
            step_1, step_2 = method

            ## ============================================================================
            ## step 1
            ## ============================================================================
            if step_1 == 'top_k_mean':
                k = num_sents_to_avg_over_per_sent[col]
                agg_1 = grouped[col].pipe(top_k_mean_agg, k=k, top_or_bottom='top')

            elif step_1 == 'bottom_k_mean':
                k = num_sents_to_avg_over_per_sent[col]
                agg_1 = grouped[col].pipe(top_k_mean_agg, k=k, top_or_bottom='bottom')

            elif step_1 == 'mean':
                agg_1 = grouped[col].mean()

            grouped_1 = agg_1.groupby(level=[0, 1])

            ## ============================================================================
            ## step 2
            ## ============================================================================
            if step_2 == 'top_k_mean':
                k = num_sents_to_avg_over_per_doc[col]
                agg = grouped_1.pipe(top_k_mean_agg, k=k, top_or_bottom='top')

            elif step_2 == 'median':
                agg = grouped_1.median()

            elif step_2 == 'mean':
                agg = grouped_1.mean()

            agg = agg.to_frame()
            aggs.append(agg)

        means = pd.concat(aggs, axis=1)
        summary_statistics.append(means)
    return pd.concat(summary_statistics)


def test_model(processed_df, model, label_df=modeling_df, num_splits=5, verbose=False):
    df = label_df.merge(processed_df.reset_index(), on=['article_url', 'press_release_url'])
    f1s = defaultdict(list)
    rocs = defaultdict(list)
    for q, y_label_col in [('q1', 'y_true_q1'), ('q2', 'y_true_q2'), ('both', 'y_true_both')]:
        if q == 'q2':
            to_train_df = df.loc[lambda df: df['y_true_q1'] == True]
        else:
            to_train_df = df

        # accuracy
        for _ in tqdm(range(num_splits), disable=not verbose):
            # model = LogisticRegressionCV(max_iter=2000, Cs=100)
            train_df, test_df = train_test_split(to_train_df, test_size=.2)
            bal_train_df = (
                train_df
                .groupby(y_label_col)
                .apply(lambda df: df.sample(train_df[y_label_col].value_counts().iloc[0], replace=True))
                .reset_index(drop=True)
            )

            model.fit(bal_train_df[['contradiction', 'entailment', 'neutral']], y=bal_train_df[y_label_col])
            y_pred = model.predict(test_df[['contradiction', 'entailment', 'neutral']])
            y_proba = model.predict_proba(test_df[['contradiction', 'entailment', 'neutral']])[:, 1]
            f1 = f1_score(test_df[y_label_col], y_pred, )
            roc = roc_auc_score(test_df[y_label_col], y_proba, )
            f1s[q].append(f1)
            rocs[q].append(roc)

    return f1s, rocs, df


# run Bayesian optimization
from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score

q_to_focus_on = 'q1'

def do_run(o_contr, o_entail, o_neutral, i_contr, i_entail, i_neutral):
    global q_to_focus_on
    param_dict = dict(
        num_sents_to_avg_over_per_doc={
            'contradiction': int(o_contr),
            'entailment': int(o_entail),
            'neutral': int(o_neutral)
        },
        num_sents_to_avg_over_per_sent={
            'contradiction': int(i_contr),
            'entailment': int(i_entail),
            'neutral': int(i_neutral)
        }
    )
    df_coref = nli_aggregation(d_coref, verbose=False, **param_dict)
    f1s, rocs, lrs, m_df = test_model(df_coref, verbose=False)
    return np.mean(pd.Series(f1s)[q_to_focus_on])


if __name__ == '__main__':
    modeling_df = pd.read_csv('cache/2024-01-26__nli-modeling-df.csv')
    modeling_df = modeling_df[['article_url', 'press_release_url', 'q1', 'q2', 'y_true_q1', 'y_true_q2', 'y_true_both']]
    d_coref = Dataset.load_from_disk('../data/s_p_500_backlinks/nli-scores-with-coref-gpt-scored-sample')

    # Run Bayesian Optimization
    params = {
        'o_contr': (10, 100),
        'o_entail': (10, 100),
        'o_neutral': (10, 100),
        'i_contr': (2, 30),
        'i_entail': (2, 30),
        'i_neutral': (2, 30),
    }

    q_to_focus_on = 'q1'
    gbm_bo = BayesianOptimization(do_run, params)
    gbm_bo.maximize(init_points=200)

    q_to_focus_on = 'q2'
    gbm_bo = BayesianOptimization(do_run, params)
    gbm_bo.maximize(init_points=200)