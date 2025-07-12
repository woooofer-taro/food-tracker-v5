import streamlit as st
import os
import pandas as pd
from datetime import datetime, timedelta
from notion_client import Client as NotionClient
from dotenv import load_dotenv
from openai import OpenAI
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# .env 読み込み
load_dotenv()

NOTION_TOKEN = os.getenv("NOTION_TOKEN")
NOTION_DATABASE_ID = os.getenv("NOTION_DATABASE_ID")
#OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# API初期化
notion = NotionClient(auth=NOTION_TOKEN)
# 旧: openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

#notionデータ取得
def fetch_notion_data(notion, database_id):
    query = notion.databases.query(
        **{
            "database_id": database_id,
            "page_size": 100,
            "filter": {
                "property": "日付",
                "date": {"is_not_empty": True}
            },
            "sorts": [{"property": "日付", "direction": "ascending"}],
        }
    )
    records = []
    for result in query["results"]:
        props = result["properties"]
        try:
            date = props["日付"]["date"]["start"]
            weight = props["体重"]["number"]
            kcal = props["kcal"]["number"]
            p = props["P"]["number"]
            f = props["F"]["number"]
            c = props["C"]["number"]
            records.append({
                "date": date,
                "weight": weight,
                "kcal": kcal,
                "P": p,
                "F": f,
                "C": c
            })
    except KeyError as e:
        print("❌ データスキップ（キーエラー）:", e)
        continue
    df = pd.DataFrame(records)
    
    # "date" が None の行は除外（←ここ追加）
    df = df[df["date"].notnull()]
    
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df

import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta

#streamlit体重グラフ
def plot_weight_chart(df):
    st.subheader("📉 体重の推移グラフ")

    period = st.selectbox("表示期間を選んでください", ["1週間", "1ヶ月", "3ヶ月", "6ヶ月", "1年"])
    today = datetime.now().date()
    delta = {
        "1週間": timedelta(days=7),
        "1ヶ月": timedelta(days=30),
        "3ヶ月": timedelta(days=90),
        "6ヶ月": timedelta(days=180),
        "1年": timedelta(days=365)
    }[period]

    start_date = today - delta
    df_filtered = df[df["date"] >= start_date]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_filtered["date"], y=df_filtered["weight"],
                             mode="lines+markers", name="体重",
                             line=dict(shape='linear')))
    fig.update_layout(
        xaxis_title="日付",
        yaxis_title="体重 (kg)",
        yaxis=dict(range=[df_filtered["weight"].min() - 2, df_filtered["weight"].max() + 2]),
        margin=dict(l=40, r=40, t=40, b=40),
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from datetime import datetime

def plot_pfc_radar_chart(df):
    st.subheader("🍗 PFCバランスの達成率（1日分）")

    selected_date = st.date_input("対象日を選択", value=datetime.now().date())

    day_data = df[df["date"] == selected_date]
    if day_data.empty:
        st.warning("選択した日にデータがありません")
        return

    actual_P = day_data["P"].sum()
    actual_F = day_data["F"].sum()
    actual_C = day_data["C"].sum()

    col1, col2, col3 = st.columns(3)
    with col1:
        target_P = st.number_input("目標P（たんぱく質）", min_value=1, value=100)
    with col2:
        target_F = st.number_input("目標F（脂質）", min_value=1, value=60)
    with col3:
        target_C = st.number_input("目標C（炭水化物）", min_value=1, value=200)

    rate_P = actual_P / target_P * 100
    rate_F = actual_F / target_F * 100
    rate_C = actual_C / target_C * 100

    labels = ["P", "F", "C"]
    values = [rate_P, rate_F, rate_C]
    values += values[:1]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, values, marker='o')
    ax.fill(angles, values, alpha=0.25)
    ax.set_yticks([0, 25, 50, 75, 100, 120])
    ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%', '120%'])
    ax.set_ylim(0, 120)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_title(f"{selected_date} のPFC達成率", size=14)

    st.pyplot(fig)

st.title("🍽️食事記録")

with st.form("entry_form"):
    date = st.date_input("日付", value=datetime.now().date())
    weight = st.number_input("体重（kg）", min_value=0.0, step=0.1)
    meal_type = st.selectbox("食事区分", ["朝食", "昼食", "夕食", "間食", "夜食"])
    meal_detail = st.text_area("食事内容（例：ご飯200g、味噌汁、焼き魚）")

    submitted = st.form_submit_button("保存")

if submitted:
    with st.spinner("ChatGPTでPFC計算中..."):
        try:
            prompt = f"""
            以下の食事内容から、おおよそのカロリーとPFC（たんぱく質P・脂質F・炭水化物C）を推定してください。
            あくまで目安で大丈夫です。
            
            出力形式：
            {{"kcal": 数値, "P": 数値, "F": 数値, "C": 数値}}
            
            例：
            食事内容: コンビニのハンバーグ弁当
            出力: {{"kcal": 720, "P": 30, "F": 28, "C": 85}}
            
            では以下の食事についても出力してください。
            食事内容: {meal_detail}
            """

            chat_response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content":(
                            "あなたは食事記録アプリのアシスタントです。"
                            "ユーザーが入力した食事内容に対して、一般的な栄養情報やWeb上で知られている数値を参考に、"
                            "精密な栄養士的判断は不要ですが、ユーザーが参考にできる程度には現実的な数値を出してください。"
                            "出力はJSON形式（例: {\"kcal\": 500, \"P\": 20, \"F\": 15, \"C\": 60}）で。"
                        )
                    },
                    {"role": "user", "content": prompt},
                ]
            )

            reply = chat_response.choices[0].message.content
            st.code(reply, language="json")

            import json, re

            # PFCの抽出関数
            def parse_response_to_pfc(response_text):
                try:
                    match = re.search(r"{.*}", response_text, re.DOTALL)
                    if match:
                        return json.loads(match.group())
                except:
                    pass

                # 数字だけ抜き出して強引に割り当て
                nums = list(map(int, re.findall(r'\d+', response_text)))
                if len(nums) >= 4:
                    return {
                        "kcal": nums[0],
                        "P": nums[1],
                        "F": nums[2],
                        "C": nums[3]
                    }
                elif len(nums) == 3:
                    return {
                        "kcal": 0,
                        "P": nums[0],
                        "F": nums[1],
                        "C": nums[2]
                    }
                else:
                    return {
                        "kcal": 0,
                        "P": 0,
                        "F": 0,
                        "C": 0
                    }

            result = parse_response_to_pfc(reply)
            cal = float(result.get("kcal", 0))
            p = float(result.get("P", 0))
            f = float(result.get("F", 0))
            c = float(result.get("C", 0))


            # Notionに送信
            notion.pages.create(
                parent={"database_id": NOTION_DATABASE_ID},
                properties={
                    "日付": {"date": {"start": str(date)}},
                    "体重": {"number": weight},
                    "区分": {"select": {"name": meal_type}},
                    "食事内容": {"title": [{"text": {"content": meal_detail}}]},
                    "カロリー": {"number": cal},
                    "P": {"number": p},
                    "F": {"number": f},
                    "C": {"number": c},
                }
            )

            st.success("✅ Notionに保存しました！")

        except Exception as e:
            st.error(f"❌ エラーが発生しました：{e}")

df = fetch_notion_data(notion, NOTION_DATABASE_ID)
plot_weight_chart(df)
plot_pfc_radar_chart(df)
