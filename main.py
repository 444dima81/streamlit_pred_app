import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

st.set_page_config(page_title="Прогноз временных рядов", layout="wide")
st.title("📈 Прогнозирование временных рядов с Prophet")

FORECAST_FILE = "forecast.csv"

# --- Загрузка файла ---
uploaded_file = st.file_uploader(
    "Загрузите CSV с временным рядом (обязательные колонки 'Order Date' и 'Sales')",
    type=["csv"]
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if not {"Order Date", "Sales"}.issubset(df.columns):
        st.error("Файл должен содержать колонки 'Order Date' и 'Sales'")
    else:
        # --- Подготовка данных ---
        data_prophet = (
            df[['Order Date','Sales']]
              .rename(columns={'Order Date':'ds','Sales':'y'})
              .assign(ds=lambda d: pd.to_datetime(d['ds']))
              .set_index('ds')
              .resample('W-MON')['y'].sum()
              .reset_index()
        )

        # --- Делим данные на train/test (80/20) ---
        split_point = int(len(data_prophet) * 0.8)
        train = data_prophet.iloc[:split_point]
        test  = data_prophet.iloc[split_point:]

        MODEL_PARAMS_FILE = "prophet_params.pkl"

        # --- Прогноз: загружаем готовый или обучаем ---
        weeks_ahead = st.number_input("Сколько недель прогнозировать вперед?", min_value=1, value=30)
        prediction_points = len(test) + weeks_ahead

        if os.path.exists(FORECAST_FILE):
            forecast = pd.read_csv(FORECAST_FILE, parse_dates=['ds'])
        else:
            model = Prophet()
            model.fit(train)
            future = model.make_future_dataframe(periods=prediction_points, freq='7D')
            forecast = model.predict(future)
            forecast.to_csv(FORECAST_FILE, index=False)

        # --- Метрики ---
        st.subheader("📊 Метрики")
        st.markdown("""
        ##### MAE (Mean Absolute Error) — средняя абсолютная ошибка, показывает насколько в среднем прогноз отклоняется от реальных значений. 
        ##### Чем меньше, тем точнее прогноз.  
        ##### R² (Коэффициент детерминации) — показывает, какая доля вариации реальных данных объясняется моделью.
        #####  Значение 1 → идеальный прогноз, 0 → модель выдаст средней реузльтат, отрицательное → модель хуже среднего
        """)
        train_pred = forecast[forecast['ds'].isin(train['ds'])]
        test_pred  = forecast[forecast['ds'].isin(test['ds'])]
        st.write(f"Train → MAE: {mean_absolute_error(train['y'], train_pred['yhat']):.2f}, R²: {r2_score(train['y'], train_pred['yhat']):.2f}")
        st.write(f"Test → MAE: {mean_absolute_error(test['y'], test_pred['yhat']):.2f}, R²: {r2_score(test['y'], test_pred['yhat']):.2f}")

        # --- Визуализация ---
        st.subheader("📈 Прогноз на всем ряде")
        st.markdown("""
        ##### Красная линия: это прогноз модели Prophet. Она строится на основе тренда и сезонности временного ряда.
        ##### Прогноз на 30 недель вперёд: красная линия продолжает ряд на 30 недель после последней доступной даты.
        ##### Вы видите, как модель ожидает изменяться продажи в будущем на основе предыдущих закономерностей.
        ##### Интервал прогноза (светло-красная область): показывает возможное отклонение прогноза,
        ##### т.е. диапазон, в котором с высокой вероятностью будут находиться реальные значения.
        """)
        fig, ax = plt.subplots(figsize=(14,6))
        ax.plot(data_prophet['ds'], data_prophet['y'], label='Исторические данные', alpha=0.6)
        ax.plot(forecast['ds'], forecast['yhat'], label='Прогноз Prophet', color='red')
        ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='red', alpha=0.2)
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        # --- Декомпозиция ---
        st.subheader("⚙️ Декомпозиция тренда и сезонностей")
        st.markdown("""
        #### 1 график
        Показывает общую направленность ряда — растут продажи со временем или падают. \n
        Линия тренда помогает понять долгосрочное движение. 
        #### 2 график 
        Отражает регулярные повторяющиеся колебания — например, рост продаж перед праздниками или летом. \n
        Это закономерности, которые повторяются через фиксированный период (неделя, месяц, год).
        """)
        model = Prophet()
        model.fit(train)
        fig2 = model.plot_components(forecast)
        st.pyplot(fig2)