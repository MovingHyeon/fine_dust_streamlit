import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn
import statsmodels

import statsmodels.api as sm
# from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin
from MaskedPCA import MaskedPCA

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
import xgboost

import pickle
import streamlit as st
import shap
import datetime

import json
import folium
import streamlit_folium
from streamlit_folium import st_folium
from folium.features import GeoJsonPopup, GeoJsonTooltip
import plotly.graph_objects as go
from streamlit_shap import st_shap

mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['font.family'] = "Malgun Gothic"
mpl.rcParams['font.size'] = 12


seoul_district = ["강남구", "강동구", "강서구", "강북구", "관악구", "광진구", "구로구", "금천구", "노원구", "동대문구", "도봉구", "동작구", "마포구", "서대문구", "성동구", "성북구", "서초구", "송파구", "영등포구", "용산구", "양천구", "은평구", "종로구", "중구", "중랑구"]

### Data

df = pd.read_csv("final_dataset_v4.csv")

df_model = df.drop(["year", "month", "district", "resid"], axis = 1)
X = df_model.drop(["PM25"], axis = 1)
y = df_model.loc[:, "PM25"]


### Model

with open('saved_model1_scaler', 'rb') as f:
    std_scaler = pickle.load(f)

with open('saved_model1_pca', 'rb') as f:
    pca = pickle.load(f)

with open('saved_model1_xgb', 'rb') as f:
    xgb = pickle.load(f)


### Applying

OPT_PC = 6
MASK = np.arange(23) >= 2

X_scaled = std_scaler.fit_transform(X)  

PCs = pca.fit_transform(X_scaled)

PCs_weights = pd.DataFrame(data = pca.components_, columns = X.columns[MASK], index = ["PC%d"%i for i in range(1, OPT_PC + 1)])

PCs = pd.DataFrame(data = PCs, columns = ["평균풍속(m/s)", "월강수량합(mm)"] + ['PC%d'%i for i in range(1, OPT_PC + 1)])

final_df = pd.concat([df.loc[:, ["year", "month", "district"]], y, PCs], axis = 1)
final_df = final_df.drop(["평균풍속(m/s)", "월강수량합(mm)"], axis = 1)
final_df = pd.concat([final_df, df.loc[:, ["평균풍속(m/s)", "월강수량합(mm)"]]], axis = 1)

y_pred = xgb.predict(PCs)
mse = np.mean((y_pred - y.to_numpy().reshape(-1))**2)


### Geo
with open("seoul_district_2017.geojson", "r", encoding = "UTF-8") as f:
    seoul_geo = json.load(f)


### Map style
def style_function(feature):
    return {
        'opacity': 0.5,
        'weight': 1,
        'color': 'white',
        'fillOpacity': 0.2,
        # 'dashArray': '5, 5',
    }





### STREAMLIT

st.set_page_config(
    page_title = "서울시 초미세먼지 대시보드",
    layout = "wide"
)

st.title("서울시 초미세먼지 Dashboard")


#

mapbox, metricbox = st.columns([0.5, 0.5])

@st.cache_resource
def date_states():
    return {"year": 2017, "month": 1}

@st.cache_resource
def district_states():
    return {"district": "강남구"}

current_date = date_states()
current_district = district_states()


##
with mapbox:

    date_year, date_month, feature_widget, _= st.columns([0.2, 0.2, 0.4, 0.2])

    year = date_year.selectbox(
        label = "연도",
        key = "map_year",
        options = (2017, 2018, 2019, 2020, 2021),
        index = current_date["year"] - 2017
    )
    month = date_month.selectbox(
        label = "월",
        key = "map_month",
        options = (1, 2, 3, 4, 5, 6,
                7, 8, 9, 10, 11, 12),
        index = current_date["month"] - 1
    )

    if year:
        current_date.update({"year": year, "month": month})

    if month:
        current_date.update({"year": year, "month": month})



    feature = feature_widget.selectbox(
        label = "지표",
        key = "map_feature",
        options = ("초미세먼지(μg/m3)", "월 강수량 합(mm)", "평균 풍속(m/s)", "E 지표", "G 지표", "O 지표", "T 지표"),
        index = 0
    )

    if feature == "초미세먼지(μg/m3)": feature = "PM25"
    elif feature == "월 강수량 합(mm)": feature = "월강수량합(mm)"
    elif feature == "평균 풍속(m/s)": feature = "평균풍속(m/s)"
    elif feature == "E 지표": feature = "PC1"
    elif feature == "G 지표": feature = "PC6"
    elif feature == "O 지표": feature = "PC5"
    elif feature == "T 지표": feature = "PC3"

    map_df = final_df.loc[final_df["year"] == year, :].loc[final_df["month"]  == month, :]
    

    map = folium.Map(
        location=[37.5651, 126.98955], 
        zoom_start=10,
        tiles='cartodb dark_matter'
    )

    tooltip = GeoJsonTooltip(
        fields=["SIG_KOR_NM"],
        aliases=["자치구"],
        localize=True,
        sticky=False,
        labels=True,
        style="""
            background-color: #F0EFEF;
            border-radius: 3px;
            box-shadow: 3px;
        """,
        max_width=800,
    )

    folium.Choropleth(
        geo_data = seoul_geo,
        data = map_df,
        columns = ['district', feature],
        fill_color = 'YlGn',
        fill_opacity = 0.5,
        line_opacity = 0.5,
        key_on = 'feature.properties.SIG_KOR_NM',
        legend_name = feature,
        highlight=True
    ).add_to(map)
    folium.GeoJson(
        data = seoul_geo,
        style_function = style_function,
        tooltip = tooltip
    ).add_to(map)

    # with st.container(height = 600, border = False):
    #     css = '''
    #     <style>
    #         [data-testid="ScrollToBottomContainer"] {
    #             overflow: hidden;
    #         }
    #     </style>
    #     '''
    #     st.markdown(css, unsafe_allow_html=True)
    #     st_folium(map, use_container_width = True, height = 600)

    st.components.v1.html(map._repr_html_(), width = 650, height = 450)


    st.bar_chart(
        data = map_df,
        x = "district",
        y = feature,
        use_container_width = True
    )

##

with metricbox:
    date_year2, date_month2, district_metric, _ = st.columns([0.2, 0.2, 0.3, 0.3])

    year2 = date_year2.selectbox(
        label = "연도",
        key = 'metric_year',
        options = (2017, 2018, 2019, 2020, 2021),
        index = current_date["year"] - 2017
    )
    month2 = date_month2.selectbox(
        label = "월",
        key = "metric_month",
        options = (1, 2, 3, 4, 5, 6,
                7, 8, 9, 10, 11, 12),
        index = current_date["month"] - 1
    )

    district2 = district_metric.selectbox(
        label = "자치구",
        key = "metrict district",
        options = ("강남구", "강동구", "강서구", "강북구", "관악구", "광진구", "구로구", "금천구", "노원구", "동대문구", "도봉구", "동작구", "마포구", "서대문구", "성동구", "성북구", "서초구", "송파구", "영등포구", "용산구", "양천구", "은평구", "종로구", "중구", "중랑구"),
        index = seoul_district.index(current_district["district"])
    )

    if year2:
        current_date.update({"year": year2, "month": month2})

    if month2:
        current_date.update({"year": year2, "month": month2})

    if district2:
        current_district.update({"district": district2})

    
    metric_df = final_df.loc[final_df["year"] == year, :].loc[final_df["month"]  == month, :].loc[final_df["district"] == district2]


    metric1, metric2= st.columns(2)
    
    with metric1:
        fig1 = go.Figure(
                go.Indicator(
                mode = "gauge+number",
                value = round(metric_df.loc[:, "PC1"].values[0], 2),
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "E 지표"},
                gauge = {
                    'bar' : {'color': "blue"},
                    'axis': {'range': [-10, 10]},
                    'steps' : [
                        {'range': [2.4, 7.2], 'color': "lightgray"},
                        {'range': [7.2, 10], 'color': "gray"}
                    ]
                },
            )
        )

        fig1.update_layout(width = 250, height=200, margin_b = 0, margin_t = 45, margin_r = 20, margin_l = 25)

        st.plotly_chart(
            fig1,
            theme = "streamlit"
        )
    with metric2:
        fig2 = go.Figure(
                go.Indicator(
                mode = "gauge+number",
                value = round(metric_df.loc[:, "PC6"].values[0], 2),
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "G 지표"},
                gauge = {
                    'bar' : {'color': "green"},
                    'axis': {'range': [-10, 10]},
                    'steps': [
                        {'range': [1.1, 3.3], 'color': 'lightgray'},
                        {'range': [3.3, 10], 'color': 'gray'}
                    ]
                }
            )
        )

        fig2.update_layout(width = 250, height=200,  margin_b = 0, margin_t = 45, margin_r = 20, margin_l = 25)

        st.plotly_chart(
            fig2,
            theme = "streamlit"
        )

    metric3, metric4= st.columns(2)
    with metric3:
        fig3 = go.Figure(
                go.Indicator(
                mode = "gauge+number",
                value = round(metric_df.loc[:, "PC5"].values[0], 2),
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "O 지표"},
                gauge = {
                    'axis': {'range': [-10, 10]},
                    'bar' : {'color': "orange"},
                    'steps': [
                        {'range': [1.5, 4.5], 'color': 'lightgray'},
                        {'range': [4.5, 10], 'color': 'gray'}
                    ]
                }
            )
        )

        fig3.update_layout(width = 250, height=200, margin_b = 0, margin_t = 45, margin_r = 20, margin_l = 25)

        st.plotly_chart(
            fig3,
            theme = "streamlit"
        )
    with metric4:
        fig4 = go.Figure(
                go.Indicator(
                mode = "gauge+number",
                value = round(metric_df.loc[:, "PC3"].values[0], 2),
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "T 지표"},
                gauge = {
                    'axis': {'range': [-10, 10]},
                    'steps': [
                        {'range': [-4.5, -1.5], 'color': 'lightgray'},
                        {'range': [-10, -4.5], 'color': 'gray'}
                    ],
                    'bar' : {'color': "violet"},
                }
            )
        )

        fig4.update_layout(width = 250, height=200, margin_b = 0, margin_t = 45, margin_r = 20, margin_l = 25)

        st.plotly_chart(
            fig4,
            theme = "streamlit"
        )

    st.markdown("**:blue[E 지표: ENERGY & CARBON]**: 전기, 수도 사용량, 탄소 배출량, 음식물폐기물 배출량 관련 지표, 석유사용량도 일정 수준 고려")
    
    st.markdown("**:green[G 지표: GAS & WASTE]**: 가스 사용량과 음식물 폐기물 재활용률 관련 대표 지표, 생할폐기물, 지정폐기물 배출량도 일정 수준 고려")

    st.markdown("**:orange[O 지표: OIL TRANSPORTS]**: 석유 사용량과 관련 교통에 관한 지표, 버스와 자차 이용할 수록 수치 증가, 전기차와 따릉이 이용할 수록 수치 감소")

    st.markdown("**:violet[T 지표: TRNASPORTS & WASYE]**: 대중교통 이용률, 생활폐기물 및 지정폐기물 배출량 관련 지표, 이 지수는 더 낮을 수록 안 좋음")

    

st.markdown("---")

st.header("자치구별 초미세먼지 요인 작용 양상")

### SHAP

# shap.initjs()

explainer = shap.Explainer(xgb)
PCs_new = pd.concat([df.loc[:, ["year", "month", "district"]], PCs], axis = 1)
PCs_district = PCs_new.loc[PCs_new["district"] == current_district["district"], :]
PCs_district["YM"] = pd.to_datetime(
    PCs_district.loc[:, "year"].astype(str) + "-" + PCs_district.loc[:, "month"].astype(str), 
    format = "%Y-%m"
)
PCs_district = PCs_district.set_index(keys = "YM")
shap_values_district = explainer(PCs_district.drop(["year", "month", "district"], axis = 1))

###

district_shap, _ = st.columns([0.15, 0.85])
district3 = district_shap.selectbox(
        label = "자치구",
        key = "shap_district",
        options = ("강남구", "강동구", "강서구", "강북구", "관악구", "광진구", "구로구", "금천구", "노원구", "동대문구", "도봉구", "동작구", "마포구", "서대문구", "성동구", "성북구", "서초구", "송파구", "영등포구", "용산구", "양천구", "은평구", "종로구", "중구", "중랑구"),
        index = seoul_district.index(current_district["district"])
    )

if district3:
    current_district.update({"district": district3})
       

st_shap(shap.plots.force(shap_values_district[0]))


# data-testid = "stVerticalBlockBorderWrapper"

# st.dataframe(df, use_container_width = True)

# col1, col2, col3 = st.columns(3)
# col1.metric(
#     label = "초미세먼지", 
#     value = round(df.loc[1, "PM25"], 1), 
#     delta = round(df.loc[1, "PM25"] - df.loc[0, "PM25"], 1)
# )
# col2.metric(
#     label = "월강수량", 
#     value = round(df.loc[1, "월강수량합(mm)"], 1), 
#     delta = round(df.loc[1, "월강수량합(mm)"] - df.loc[0, "월강수량합(mm)"], 1)
# )

# col1, col2 = st.columns(2)
# col1.metric(
#     label = "초미세먼지", 
#     value = round(df.loc[1, "PM25"], 1), 
#     delta = round(df.loc[1, "PM25"] - df.loc[0, "PM25"], 1)
# )
# col2.metric(
#     label = "월강수량", 
#     value = round(df.loc[1, "월강수량합(mm)"], 1), 
#     delta = round(df.loc[1, "월강수량합(mm)"] - df.loc[0, "월강수량합(mm)"], 1)
# )

# # checkbox/radio(선택지)/selectbox(밑으로 내려가는 선택지, 기본값은 index=으로 설정)/mutliselect(다중선택지)/slider/text_input/number_input 버튼도 있음
# button = st.button("눌러봐")
# if button:
#     st.write(":orange[버튼]이 눌렸습니다")

# button2 = st.download_button(
#     label = "데이터 다운로드(csv)",
#     data = df.to_csv(encoding = "EUC-KR", index = False),
#     file_name = "서울시_초미세먼지_관련_데이터.csv",
#     mime = "text/csv"
# )

# statement = st.text_input(
#     label = "물어보고 싶은 것이 있나요?",
#     placeholder = "여기에 물어보세요",
#     autocomplete = "hi"
# )

# st.subheader("4대 초미세먼지 지표 설명")

# # blue, green, orange, red, violet
# st.markdown("**:green[PC6]**: 지정폐기물과 생활폐기물 관련 대표 지표, 가스 사용량도 고려")

# st.markdown("**:green[PC5]**: 자가용이나 대중교통 대신 따릉이 이용을 나타내는 지표, 공원 면적과 건설폐기물의 재활용률도 고려")

# st.markdown("**:green[PC5]**: 자가용이나 대중교통 대신 따릉이 이용을 나타내는 지표, 공원 면적과 건설폐기물의 재활용률도 고려")
