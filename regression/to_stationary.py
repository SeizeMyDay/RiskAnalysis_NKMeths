import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"

HKexport = pd.read_csv('HK_exports_to_korea_preprocessed.csv')
KRseizure = pd.read_csv('KR_MethSeizure_preprocessed.csv')

df = pd.merge(KRseizure, HKexport, left_on='date', right_on='Year').drop(columns=['Year'])


# 독립변수: HK_exports_to_Korea
# 원본 독립변수: 정상성 불만족, 이분산성 존재
# 로그, 변화율 둘 다 적용: 변화율만 적용해도 정상성 만족하나 이분산성 해소 위해 로그 적용.

# 로그, 변화율 적용 시 정상성 만족 확인
from statsmodels.tsa.stattools import adfuller
col = 'HK_export_to_Korea'
result = adfuller(np.log(df[col]).diff().iloc[1:])
print(f'Column: {col}')
print(f'Test statistic: {result[0]}')
print(f'p-value: {result[1]}')
if result[1] < 0.05:
    print('The time series is stationary')
else:
    print('The time series is non-stationary')
print()

# 쓸 독립변수: 로그 취하고 차분
df['HK_exports_to_Korea_logdiff'] = np.log(df['HK_export_to_Korea']).diff()
df.drop(columns='HK_export_to_Korea', inplace=True)

# 전처리 완료된 독립변수(HK_exports_to_Korea_logdiff) 시각화
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['date'], y=df['HK_exports_to_Korea_logdiff'], mode='lines'))
fig.update_layout(title_text='HK_exports_to_Korea_logdiff', title_x=0.5, 
                  width=1200, height=400)
fig.show()


# 종속변수: seizure
# 원본 종속변수: 정상성 불만족
# 변화율만 적용: 추세가 뚜렷하지 않으므로 로그까지 적용할 필요는 없다고 판단했음.

# 변화율 적용 시 정상성 만족 확인
from statsmodels.tsa.stattools import adfuller
col='seizure'
result = adfuller(df[col].diff().iloc[1:])
print(f'Column: {col}')
print(f'Test statistic: {result[0]}')
print(f'p-value: {result[1]}')
if result[1] < 0.05:
    print('The time series is stationary')
else:
    print('The time series is non-stationary')
print()

df['seizure_diff'] = df['seizure'].diff()
df.drop(columns='seizure', inplace=True)

# 전처리 완료된 종속변수(seizure_diff) 시각화
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['date'], y=df['seizure_diff'], mode='lines'))
fig.update_layout(title_text='seizure_diff', title_x=0.5, 
                  width=1200, height=400)
fig.show()

# 결과 저장 -> regression_최종.py에서 활용
df.to_csv('regression.csv', index=False)