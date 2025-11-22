!pip install category_encoders

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_squared_log_error, r2_score, mean_absolute_error
import category_encoders as ce
import xgboost as xgb
import missingno as msno
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_log_error, r2_score, mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

RANDOM_STATE = 42

df = pd.read_csv("/content/all_v2.csv")

df

df.info(verbose = True, show_counts = True)

df.isna().sum()

print("\nОписание числовых признаков:")
display(df.describe().T)

df.duplicated().sum()

df = df.drop_duplicates()

df.shape

df = df[df['region']==2661]
print("После фильтра по региону:", df.shape)

mask = (
    df['area'].between(20,200) &
    df['kitchen_area'].between(6,30) &
    df['price'].between(1_500_000,50_000_000)
)

df = df[mask]
print("После фильтра типовых объектов:", df.shape)

df['rooms'] = df['rooms'].replace({-1:0, -2:0})

df

df['dt'] = pd.to_datetime(df['date'] + ' ' + df['time'])

df['floor_ratio'] = df['level'] / df['levels']
df['is_first']   = (df['level']==1).astype(int)
df['is_last']    = (df['level']==df['levels']).astype(int)

df['area_ratio'] = np.where(
    df['rooms']== 0,
    df['area'],
    df['kitchen_area']/df['area']
)

ref = pd.to_datetime('2025-05-20')
df['ad_age_days'] = (ref - df['dt']).dt.days

df

corr = df.select_dtypes(include='number').corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr, cmap='coolwarm', center=0, fmt='.2f')
plt.title("Корреляционная матрица")
plt.show()

drop_cols = ['price','date','time','dt']

df_model = df.drop(columns=drop_cols)
y = df['price']

df_model.shape

cat_cols = df_model.select_dtypes(include='object').columns.tolist()
encoder = ce.BinaryEncoder(cols=cat_cols)
df_enc = encoder.fit_transform(df_model)

# Формируем X и y
X = df_enc
y = np.log1p(df['price'])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

num_cols = X.select_dtypes(include='number').columns.tolist()
scaler = StandardScaler()
X_train_s = X_train.copy()
X_test_s  = X_test.copy()
X_train_s[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test_s[num_cols]  = scaler.transform(X_test[num_cols])

ridge = Ridge(alpha=1.0, random_state=42)
ridge.fit(X_train_s, y_train)
pred_tr_r = ridge.predict(X_train_s)
pred_te_r = ridge.predict(X_test_s)

xgb_model = xgb.XGBRegressor(random_state=42, use_label_encoder=False, eval_metric='rmse')
xgb_model.fit(X_train, y_train)

pred_tr_x = xgb_model.predict(X_train)
pred_te_x = xgb_model.predict(X_test)

def calc_metrics(y_true, y_pred):
    return {
        'RMSLE': np.sqrt(mean_squared_log_error(y_true, np.maximum(y_pred,0))),
        'R2':    r2_score(y_true, y_pred),
        'MAE':   mean_absolute_error(y_true, y_pred)
    }


results = []
for name, y_tr, p_tr, y_te, p_te in [
    ('Ridge',    y_train, pred_tr_r, y_test, pred_te_r),
    ('XGBoost',  y_train, pred_tr_x, y_test, pred_te_x),
]:
    m_tr = calc_metrics(y_tr, p_tr)
    m_te = calc_metrics(y_te, p_te)
    results.append({'Model': name, 'Set': 'Train', **m_tr})
    results.append({'Model': name, 'Set': 'Test',  **m_te})
results_df = pd.DataFrame(results)
display(results_df)

coef = pd.Series(np.abs(ridge.coef_), index=X_train_s.columns)
print("Top-10 признаков Ridge:")
display(coef.sort_values(ascending=False).head(10))

imp = pd.Series(xgb_model.feature_importances_, index=X_train.columns)
print("Top-10 признаков XGBoost:")
display(imp.sort_values(ascending=False).head(10))

new = pd.DataFrame([{
    'date':          '2025-05-20',
    'time':          '12:00:00',
    'geo_lat':       59.83,
    'geo_lon':       30.26,
    'region':        2661,
    'building_type': 1,
    'level':         3,
    'levels':        5,
    'rooms':         1,
    'area':          25.0,
    'kitchen_area':  7.0,
    'object_type':   1
}])
new['dt']          = pd.to_datetime(new['date'] + ' ' + new['time'])
new['rooms']       = new['rooms'].replace({-1:0, -2:0})
new['floor_ratio'] = new['level'] / new['levels']
new['is_first']    = (new['level']==1).astype(int)
new['is_last']     = (new['level']==new['levels']).astype(int)
new['area_ratio']  = np.where(new['rooms']==0, new['area'], new['kitchen_area']/new['area'])
new['ad_age_days'] = (ref - new['dt']).dt.days
# Собираем new_model по df_model.columns
model_cols = df_model.columns
new_model  = new[model_cols].fillna(0)
# Кодирование
new_enc = encoder.transform(new_model) if encoder else new_model.copy()
new_X   = new_enc.reindex(columns=X_train.columns, fill_value=0)
# Стандартизация для Ridge
new_X_s = new_X.copy()
new_X_s[num_cols] = scaler.transform(new_X[num_cols])
# Предсказания
pred_ridge = ridge.predict(new_X_s)[0]
pred_xgb   = xgb_model.predict(new_X)[0]
print(f"Predicted price Ridge: {np.expm1(pred_ridge):,.0f} ₽")
print(f"Predicted price XGBoost: {np.expm1(pred_xgb):,.0f} ₽")

rent = pd.read_csv('move.csv')

rent

rent.info(verbose = True, show_counts = True)

print("\nЧисло пропусков по столбцам:")
display(rent.isna().sum())

print("\nОписание числовых признаков:")
display(rent.describe().T)

numeric_cols = ['price','total_area','living_area','kitchen_area','storey','storeys','fee_percent','views','minutes']
corr_rent = rent[numeric_cols].corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr_rent, annot=True, fmt=".2f", cmap='coolwarm', center=0)
plt.title("Корреляционная матрица rent")
plt.show()

rent['total_area'] = (
    rent['total_area'].astype(float)
)

rent['provider'] = (
    rent['provider']
    .astype(str)
    .str.strip()
    .str.title()
)
rent['metro'] = (
    rent['metro']
    .astype(str)
    .str.strip()
    .str.title()
)

rent['provider']

dups = rent.duplicated().sum()
print(f"Найдено дубликатов: {dups}")

# Список ключевых числовых колонок для проверки
numeric_cols = ['price', 'storeys', 'total_area', 'living_area', 'kitchen_area']

# Рассчитаем Z-оценки и визуализируем для каждой
for col in numeric_cols:
    rent[f'z_{col}'] = zscore(rent[col].fillna(rent[col].mean()))
    plt.figure(figsize=(6,3))
    sns.histplot(rent[f'z_{col}'], bins=50, kde=True)
    plt.title(f"Z-распределение для {col}")
    plt.xlabel(f"z_{col}")
    plt.tight_layout()
    plt.show()

# Оставляем только объекты, у которых |z| < 3 по всем выбранным колонкам
mask = np.ones(len(rent), dtype=bool)
for col in numeric_cols:
    mask &= rent[f'z_{col}'].abs() < 3


print("До фильтрации:", rent.shape)
rent = rent[mask]
print("После фильтрации (|z|<3 по всем):", rent.shape)

# Удалим вспомогательные столбцы с Z-оценками
rent.drop(columns=[f'z_{col}' for col in numeric_cols], inplace=True)

# Формируем X и y
Xr = rent.drop(columns=['price'])
yr = rent['price']

# One-Hot кодирование всех категориальных: provider, metro, way
Xr = pd.get_dummies(Xr, columns=['provider','metro','way'], drop_first=True)

Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(
    Xr, yr, test_size=0.2, random_state=42
)

models = {
    'RandomForest': RandomForestRegressor(random_state=42, n_jobs=-1),
    'XGBoost':      xgb.XGBRegressor(random_state=42, use_label_encoder=False, eval_metric='rmse'),
    'ExtraTrees':   ExtraTreesRegressor(random_state=42, n_jobs=-1),
}

for name, model in models.items():
    model.fit(Xr_tr, yr_tr)
    p_tr = model.predict(Xr_tr)
    p_te = model.predict(Xr_te)
    print(f"\n{name}:")
    print(f"  Train RMSLE: {np.sqrt(mean_squared_log_error(yr_tr, np.maximum(p_tr,0))):.4f}")
    print(f"  Test  RMSLE: {np.sqrt(mean_squared_log_error(yr_te, np.maximum(p_te,0))):.4f}")
    print(f"  Train R2:    {r2_score(yr_tr, p_tr):.4f}")
    print(f"  Test  R2:    {r2_score(yr_te, p_te):.4f}")
    print(f"  Train MAE:   {mean_absolute_error(yr_tr, p_tr):.0f}")
    print(f"  Test  MAE:   {mean_absolute_error(yr_te, p_te):.0f}")

best = max(models.keys(), key=lambda n: r2_score(yr_te, models[n].predict(Xr_te)))
print("Лучшая модель:", best)

fi_series = pd.Series(models['ExtraTrees'].feature_importances_, index=Xr_tr.columns)
grouped = (fi_series
           .groupby(lambda name: name.split('_')[0])
           .sum()
           .sort_values(ascending=False))

print("Top-10 признаков:")
display(grouped.head(10))