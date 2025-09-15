import pandas as pd
import numpy as np
from statsmodels.tools import add_constant
import statsmodels.api as sm
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"
from tqdm import tqdm


# 데이터 불러오기
df = pd.read_csv('regression.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)
df.drop(0, axis=0, inplace=True)
df.head()


# 분석 코드
# sliding window size: 24개월
# max lag: 12개월
def regress(df, date_col_name='date', x_col_name='HK_exports_to_Korea_logdiff', y_col_name='seizure_diff', window_size=24, max_lag=12) :
    
    results = []
    
    for start in tqdm(range(len(df) - window_size + 1), desc='regression') :
        end = start + window_size
        window_df = df.iloc[start: end].copy()
    
        best_r2 = -np.inf
        best_lag = None
        best_beta = None
        best_pval = None
    
        # 최적 lag 도출
        for lag in range(0, max_lag + 1) :
            X = window_df[x_col_name].shift(lag)
            y = window_df[y_col_name]
    
            valid_idx = X.notna() & y.notna()
            X_valid = X[valid_idx]
            y_valid = y[valid_idx]
    
            if len(X_valid) < 10 :
                continue
    
            X_valid = sm.add_constant(X_valid)
            
            # OLS, Newey–West
            model = sm.OLS(y_valid, X_valid).fit(
                cov_type='HAC', 
                cov_kwds={'maxlags': lag if lag > 0 else 1}  # 최소 1
            )
    
            if model.rsquared > best_r2 :
                best_r2 = model.rsquared
                best_lag = lag
                best_beta = model.params.iloc[1]
                best_pval = model.pvalues.iloc[1]
    
        # 위험도 계산, Pvalue 필터링
        if (best_beta is not None) and (best_beta > 0) and (best_pval < 0.05) :
            risk_score = best_beta
        else:
            risk_score = 0
    
        # 윈도우 중앙 시점 기준 기록
        mid_date = window_df[date_col_name].iloc[window_size // 2]
        results.append({
            'date': mid_date,
            'best_lag': best_lag,
            'beta': best_beta,
            'pval': best_pval,
            'r2': best_r2,
            'risk_score': risk_score
        })

    # 결과 도출
    result_df = pd.DataFrame(results)
    
    # risk score에 로그 적용(극단값 방지)
    result_df.loc[result_df['risk_score']!=0, 'risk_score'] = np.log(result_df.loc[result_df['risk_score']!=0]['risk_score'])
    
    return result_df


# 분석 실행
result = regress(df, date_col_name='date', x_col_name='HK_exports_to_Korea_logdiff', y_col_name='seizure_diff', window_size=24, max_lag=12)
print('\nresult')
print(result)

# 결과 저장
result[['date', 'best_lag', 'risk_score']].loc[result['risk_score'] != 0].to_csv('result.csv')

# 시각화
fig = go.Figure()
fig.add_trace(go.Scatter(x=result['date'], y=result['risk_score'], mode='lines', line={'color': 'black'}))
fig.update_layout(title_text='Risk Score over Time', title_x=0.5,  
                  width=1200, height=300, 
                  margin_t=50, margin_b=10, margin_l=0, margin_r=0, 
                  font={'size': 20, 'color': "black"})
fig.show()