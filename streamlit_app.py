#!/usr/bin/env python3
"""
行政手続等オンライン化状況 データ可視化・分析ダッシュボード

日本の法令に基づく行政手続等のオンライン化状況を可視化・分析する
Streamlitベースの対話的データ分析ツール
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import re
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
import warnings

warnings.filterwarnings('ignore')

# ページ設定
st.set_page_config(
    page_title="行政手続等オンライン化状況ダッシュボード",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# データファイルのパス
DATA_DIR = Path(__file__).parent / "docs"
CSV_FILE = DATA_DIR / "20250729_procedures-survey-results_outline_02.csv"
PARQUET_FILE = DATA_DIR / "procedures_data.parquet"

# CSVのカラム定義
COLUMNS = [
    "手続ID",
    "所管府省庁", 
    "手続名",
    "法令名",
    "法令番号",
    "根拠条項号",
    "手続類型",
    "手続主体",
    "手続の受け手",
    "経由機関",
    "独立行政法人等の名称",
    "事務区分",
    "府省共通手続",
    "実施府省庁",
    "オンライン化の実施状況",
    "オンライン化の実施予定及び検討時の懸念点",
    "オンライン化実施時期",
    "申請等における本人確認手法",
    "手数料等の納付有無",
    "手数料等の納付方法",
    "手数料等のオンライン納付時の優遇措置",
    "処理期間(オンライン)",
    "処理期間(非オンライン)",
    "情報システム(申請)",
    "情報システム(事務処理)",
    "総手続件数",
    "オンライン手続件数",
    "非オンライン手続件数",
    "申請書等に記載させる情報",
    "申請時に添付させる書類",
    "添付書類等提出の撤廃/省略状況",
    "添付書類等の提出方法",
    "添付書類等への電子署名",
    "添付形式等が定められた規定",
    "手続が行われるイベント(個人)",
    "手続が行われるイベント(法人)",
    "申請に関連する士業",
    "申請を提出する機関"
]

# --- 項目定義（要約） & 表示順 ---
FIELD_DEFS: Dict[str, str] = {
    "所管府省庁": "手続の根拠法令（条文）を所管する府省庁。",
    "手続名": "手続の名称。",
    "法令名": "手続の根拠となる法令の正式名称。",
    "法令番号": "根拠法令の法令番号。",
    "根拠条項号": "根拠条・項・号の番号。",
    "手続類型": "1申請等 / 2-1申請等に基づく処分通知等 / 2-2申請等に基づかない処分通知等 / 2-3交付等(民間手続) / 3縦覧等 / 4作成・保存等。",
    "手続主体": "手続を行う主体（国、独立行政法人等、地方等、国民等、民間事業者等 等の組合せを含む）。",
    "手続の受け手": "申請等において最終的に手続を受ける者（国、独立行政法人等、地方等、国民等、民間事業者等 等）。",
    "経由機関": "法令に基づき申請等の提出時に経由が必要な機関の種別。",
    "事務区分": "地方公共団体が行う事務の区分（自治事務 / 第1号法定受託事務 / 第2号法定受託事務 / 地方の事務でない）。",
    "府省共通手続": "全府省共通(○) / 一部府省共通(●) / 非共通(×)。",
    "実施府省庁": "当該手続を実施する府省庁（府省共通手続は全回答を列挙）。",
    "オンライン化の実施状況": "1実施済 / 2未実施 / 3適用除外 / 4その他 / 5一部実施済。",
    "オンライン化の実施予定及び検討時の懸念点": "予定または検討時の懸念（制度改正、システム未整備、原本紙等）。",
    "オンライン化実施時期": "オンライン化の実施予定年度（2024〜2030以降）。",
    "申請等における本人確認手法": "押印＋印鑑証明 / 押印 / 署名 / 本人確認書類提示・提出 / その他 / 不要。",
    "手数料等の納付有無": "手数料等の有無。",
    "手数料等の納付方法": "オフライン（窓口/銀行/ATM/コンビニ等）・オンライン（ペイジー/クレカ/QR等）。",
    "手数料等のオンライン納付時の優遇措置": "オンライン納付による減免の有無。",
    "処理期間(オンライン)": "オンライン手続の標準処理期間。",
    "処理期間(非オンライン)": "非オンライン手続の標準処理期間。",
    "情報システム(申請)": "申請等に係るシステム名（受付/申請）。",
    "情報システム(事務処理)": "申請等を受けた後の事務処理システム名。",
    "総手続件数": "令和5年度等の年間総件数（有効数字2桁目安、試算含む）。",
    "オンライン手続件数": "オンラインで実施した件数（該当手続のみ）。",
    "非オンライン手続件数": "オンライン以外で実施した件数。",
    "申請書等に記載させる情報": "申請書記入の必須項目（マイナンバー、法人番号等）。",
    "申請時に添付させる書類": "申請時に提出が必須の典型書類（住民票、戸籍、登記事項等）。",
    "添付書類等提出の撤廃/省略状況": "添付書類撤廃・省略の状況（済/予定/不可/その他）。",
    "添付書類等の提出方法": "電子/原紙/一部電子等の提出方式。",
    "添付書類等への電子署名": "添付書類の電子署名の要否（不要/一部/全て）。",
    "添付形式等が定められた規定": "法令/告示/システム仕様等の規定有無。",
    "手続が行われるイベント(個人)": "個人のライフイベント（妊娠、出生、引越し、就職・転職、税金、年金、死亡・相続 等）。",
    "手続が行われるイベント(法人)": "法人のライフイベント（設立、役員変更、採用・退職、入札・契約、移転、合併・廃業 等）。",
    "申請に関連する士業": "代理申請が可能な士業（弁護士、司法書士、行政書士、税理士、社労士、公認会計士、弁理士 等）。",
    "申請を提出する機関": "提出先機関（本府省庁/出先機関/地方公共団体 等）。",
}

OPTION_ORDERS: Dict[str, List[str]] = {
    # 見やすさのための表示順（存在しない値はそのまま末尾に）
    "手続類型": [
        "申請等", "申請等に基づく処分通知等", "申請等に基づかない処分通知等",
        "交付等（民間手続）", "縦覧等", "作成・保存等"
    ],
    "オンライン化の実施状況": ["実施済", "一部実施済", "未実施", "適用除外", "その他"],
    "手続主体": ["国", "独立行政法人等", "地方等", "国又は独立行政法人等", "独立行政法人等又は地方等", "国又は地方等", "国、独立行政法人等又は地方等", "国民等", "民間事業者等", "国民等、民間事業者等"],
    "手続の受け手": ["国", "独立行政法人等", "地方等", "国又は独立行政法人等", "独立行政法人等又は地方等", "国又は地方等", "国、独立行政法人等又は地方等", "国民等", "民間事業者等", "国民等、民間事業者等"],
    "事務区分": ["自治事務", "第1号法定受託事務", "第2号法定受託事務", "地方の事務でない"],
    "府省共通手続": ["○（全府省）", "●（一部の府省）", "×（府省共通手続でない)"]
}

# ---- 正規化ユーティリティ ----
def _normalize_label(key: str, val: Any) -> str:
    s = str(val).strip()
    if s.lower() == 'nan' or s == '':
        return s
    # 統一：半角括弧→全角括弧
    s = s.replace('(', '（').replace(')', '）')
    # 先頭の分類コード（例: 1 / 2-1 / 2-3 等）を除去
    if key in ("オンライン化の実施状況", "手続類型"):
        s = re.sub(r"^\s*\d+(?:-\d+)?\s*", "", s)
    # よくある表記ゆれの吸収
    if key == "手続類型":
        # 「交付等（民間手続）」の表記ゆれ
        s = s.replace("交付等(民間手続)", "交付等（民間手続）")
    return s

def normalized_counts(df: pd.DataFrame, column: str, key: str) -> pd.Series:
    if column not in df.columns or len(df) == 0:
        return pd.Series(dtype=int)
    series = df[column].dropna().map(lambda v: _normalize_label(key, v))
    vc = series.value_counts()
    order = OPTION_ORDERS.get(key)
    if order:
        ordered = vc.reindex([v for v in order if v in vc.index]).dropna()
        # もし順序適用で空になったら、素のカウントにフォールバック
        if len(ordered) > 0:
            return ordered
    return vc

def order_series_by_option(series: pd.Series, key: str) -> pd.Series:
    order = OPTION_ORDERS.get(key)
    if not order:
        return series
    # dict for order index
    idx = {v: i for i, v in enumerate(order)}
    return series.sort_index(key=lambda s: s.map(lambda x: idx.get(x, len(idx))))

@st.cache_data(ttl=3600, show_spinner="データを読み込んでいます...")
def load_data() -> pd.DataFrame:
    """Parquetファイルからデータを高速読み込み（CSVファイルがない場合は変換）"""
    
    # Parquetファイルが存在しない場合はCSVから変換
    if not PARQUET_FILE.exists() and CSV_FILE.exists():
        st.info("初回起動：CSVファイルをParquet形式に変換しています...")
        
        # CSVファイルを読み込み
        df = pd.read_csv(
            CSV_FILE,
            encoding='utf-8-sig',
            skiprows=2,
            names=COLUMNS,
            dtype=str,
            na_values=['', 'NaN', 'nan'],
            low_memory=False
        )
        
        # カテゴリ型に変換
        categorical_cols = ['所管府省庁', '手続類型', '手続主体', '手続の受け手', 
                          'オンライン化の実施状況', '事務区分', '府省共通手続']
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype('category')
        
        # 数値型に変換
        numeric_columns = ["総手続件数", "オンライン手続件数", "非オンライン手続件数"]
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype('int32')
        
        # オンライン化率を計算
        if '総手続件数' in df.columns and 'オンライン手続件数' in df.columns:
            df['オンライン化率'] = np.where(
                df['総手続件数'] > 0,
                (df['オンライン手続件数'] / df['総手続件数'] * 100).round(2),
                0
            ).astype('float32')
        
        # Parquetファイルとして保存
        df.to_parquet(PARQUET_FILE, engine='pyarrow', compression='snappy')
        st.success("変換完了！次回からは高速に読み込めます。")
    
    # Parquetファイルから読み込み（超高速）
    df = pd.read_parquet(PARQUET_FILE, engine='pyarrow')
    
    # カテゴリ型が維持されているか確認
    categorical_cols = ['所管府省庁', '手続類型', '手続主体', '手続の受け手', 
                      'オンライン化の実施状況', '事務区分', '府省共通手続']
    for col in categorical_cols:
        if col in df.columns and df[col].dtype != 'category':
            df[col] = df[col].astype('category')
    
    # オンライン化率がない場合は計算
    if 'オンライン化率' not in df.columns:
        if '総手続件数' in df.columns and 'オンライン手続件数' in df.columns:
            df['オンライン化率'] = np.where(
                df['総手続件数'] > 0,
                (df['オンライン手続件数'] / df['総手続件数'] * 100).round(2),
                0
            ).astype('float32')
    
    return df

# セッション状態の初期化
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.df = None

@st.cache_data
def get_unique_values(df, column):
    """カラムのユニーク値を取得（キャッシュ）"""
    if df[column].dtype.name == 'category':
        # カテゴリ型の場合は高速処理
        return sorted([str(v) for v in df[column].cat.categories if pd.notna(v)])
    else:
        unique_vals = df[column].dropna().unique()
        # 全て文字列に変換してからソート
        return sorted([str(v) for v in unique_vals])

@st.cache_data
def filter_dataframe(df, ministries, statuses, types, recipients, actors=None, receivers=None, office_types=None, is_common=None, count_ranges=None):
    """データフレームをフィルタリング（キャッシュ）"""
    mask = pd.Series([True] * len(df), index=df.index)
    if ministries:
        mask &= df['所管府省庁'].isin(ministries)
    if statuses:
        mask &= df['オンライン化の実施状況'].isin(statuses)
    if types:
        mask &= df['手続類型'].isin(types)
    if recipients:
        mask &= df['手続の受け手'].isin(recipients)
    if actors:
        mask &= df['手続主体'].isin(actors)
    if receivers:
        mask &= df['手続の受け手'].isin(receivers)
    if office_types:
        mask &= df['事務区分'].isin(office_types)
    if is_common:
        mask &= df['府省共通手続'].isin(is_common)
    
    # 手続件数範囲フィルター
    if count_ranges:
        count_mask = pd.Series([False] * len(df), index=df.index)
        for range_str in count_ranges:
            if range_str == "100万件以上":
                count_mask |= df['総手続件数'] >= 1000000
            elif range_str == "10万件以上100万件未満":
                count_mask |= (df['総手続件数'] >= 100000) & (df['総手続件数'] < 1000000)
            elif range_str == "1万件以上10万件未満":
                count_mask |= (df['総手続件数'] >= 10000) & (df['総手続件数'] < 100000)
            elif range_str == "1000件以上1万件未満":
                count_mask |= (df['総手続件数'] >= 1000) & (df['総手続件数'] < 10000)
            elif range_str == "100件以上1000件未満":
                count_mask |= (df['総手続件数'] >= 100) & (df['総手続件数'] < 1000)
            elif range_str == "10件以上100件未満":
                count_mask |= (df['総手続件数'] >= 10) & (df['総手続件数'] < 100)
            elif range_str == "1件以上10件未満":
                count_mask |= (df['総手続件数'] >= 1) & (df['総手続件数'] < 10)
            elif range_str == "0件もしくは不明":
                count_mask |= (df['総手続件数'] == 0) | df['総手続件数'].isna()
        mask &= count_mask
    
    return df[mask]



# CSVエクスポート用キャッシュヘルパー
@st.cache_data
def df_to_csv_bytes(df: pd.DataFrame, columns: List[str] | None = None) -> bytes:
    if columns:
        df = df[columns]
    return df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')

# ---- Network utils ----
def _has_cols(df: pd.DataFrame, cols: List[str]) -> bool:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        st.warning("必要なカラムが見つかりません: " + ", ".join(missing))
        return False
    return True

def _safe_notna(series: pd.Series) -> pd.Series:
    # category/object列で 'nan' 文字列が混じるケースを吸収
    s = series.astype(str)
    return (s.str.lower() != 'nan') & (s.str.strip() != '')

def _layout_for_graph(G: nx.Graph):
    n = G.number_of_nodes()
    if n == 0:
        return {}
    if n <= 30:
        return nx.kamada_kawai_layout(G)
    # spring_layout は重いので反復を抑える & 固定seed
    return nx.spring_layout(G, k=1, iterations=30, seed=42)


# ---- PyVis renderer for interactive network visualization ----

def _render_pyvis(G: nx.Graph, height: int = 700):
    """Render a draggable/zoomable network with PyVis inside Streamlit."""
    net = Network(height=f"{height}px", width="100%", bgcolor="#ffffff", font_color="#333", cdn_resources='in_line')
    # Enable physics for interactive layout
    net.barnes_hut(spring_length=120, damping=0.8)

    for n, data in G.nodes(data=True):
        label = str(n)
        title = data.get('tooltip', label)
        size = max(6, int(data.get('size', 10)))
        color = data.get('color')
        if color is None:
            cat = data.get('category')
            if cat == 'personal':
                color = '#2ca02c'
            elif cat == 'corporate':
                color = '#d62728'
            elif cat == 'procedure':
                color = '#9467bd'
        net.add_node(str(n), label=label, title=title, size=size, color=color)

    for u, v, d in G.edges(data=True):
        w = int(d.get('weight', 1))
        net.add_edge(str(u), str(v), value=w, title=f"weight: {w}")

    html = net.generate_html(notebook=False)
    components.html(html, height=height, scrolling=True)

# ---- Advanced network helpers (metrics, normalization, export) ----
def _compute_centrality(G: nx.Graph, metric: str = 'degree') -> Dict[Any, float]:
    if G.number_of_nodes() == 0:
        return {}
    metric = metric.lower()
    if metric == 'degree':
        return {n: float(d) for n, d in G.degree()}
    if metric == 'betweenness':
        return nx.betweenness_centrality(G, normalized=True)
    if metric == 'eigenvector':
        try:
            return nx.eigenvector_centrality(G, max_iter=500)
        except nx.PowerIterationFailedConvergence:
            return nx.betweenness_centrality(G, normalized=True)
    if metric == 'pagerank':
        return nx.pagerank(G)
    # default
    return {n: float(d) for n, d in G.degree()}

def _scale_sizes(values: Dict[Any, float], min_size: int = 8, max_size: int = 40) -> Dict[Any, int]:
    if not values:
        return {}
    v = np.array(list(values.values()), dtype=float)
    v = (v - v.min()) / (np.ptp(v) + 1e-9)  # 0-1
    return {k: int(min_size + (max_size - min_size) * vv) for k, vv in zip(values.keys(), v)}

def _export_nodes_edges(G: nx.Graph) -> tuple[pd.DataFrame, pd.DataFrame]:
    nodes = []
    for n, d in G.nodes(data=True):
        nodes.append({'id': n, **{k: v for k, v in d.items() if k not in ['pos']}})
    edges = []
    for u, v, d in G.edges(data=True):
        rec = {'source': u, 'target': v}
        rec.update({k: v for k, v in d.items()})
        edges.append(rec)
    return pd.DataFrame(nodes), pd.DataFrame(edges)


def _cosine_normalized_weight(n_xy: int, n_x: int, n_y: int) -> float:
    # Safer than PMI for sparse small samples; 0..1 range roughly
    if n_x == 0 or n_y == 0:
        return 0.0
    return n_xy / np.sqrt(n_x * n_y)

# ---- Sankeyラベル改行ヘルパ ----
def _wrap_label(text: Any, width: int = 10, max_lines: int = 3) -> str:
    """Wrap long (JP) labels with newlines so Sankey node text doesn't overlap.
    width: number of characters per line (approx). max_lines: cap lines; add ellipsis when truncated.
    """
    if text is None:
        return ""
    s = str(text)
    if width <= 0:
        return s
    lines = [s[i:i+width] for i in range(0, len(s), width)]
    if max_lines and len(lines) > max_lines:
        lines = lines[:max_lines]
        if not lines[-1].endswith('…'):
            lines[-1] = lines[-1] + '…'
    return "\n".join(lines)

# ---- Multi-value splitter for JP list-like fields ----
def _split_multi_values(val: Any) -> List[str]:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return []
    s = str(val).strip()
    if s.lower() == 'nan' or s == '':
        return []
    for sep in ['\n', '、', ',', '，', ';', '；', '・', '/', '／']:
        s = s.replace(sep, '、')
    return [e.strip() for e in s.split('、') if e.strip()]

# --- Top-N + その他 helper ---
def _topn_with_other(series: pd.Series, top: int = 8, other_label: str = 'その他') -> pd.DataFrame:
    """Return a DataFrame with columns [label, 件数] limited to top-
    categories and aggregate the rest into 'その他'. The first column name will be inferred later."""
    vc = series.value_counts()
    dfv = vc.reset_index()
    dfv.columns = ['label', '件数']
    if len(dfv) > top:
        keep = top - 1 if top >= 2 else 1
        top_df = dfv.iloc[:keep].copy()
        other_sum = dfv.iloc[keep:]['件数'].sum()
        other_row = pd.DataFrame({'label': [other_label], '件数': [other_sum]})
        dfv = pd.concat([top_df, other_row], ignore_index=True)
    return dfv

# ---- 手続詳細ビューの描画ヘルパ ----
def _render_procedure_detail(proc_id: str, df: pd.DataFrame):
    """選択した手続IDの詳細を表示（全項目表示版）"""
    row = df[df['手続ID'] == proc_id]
    if row.empty:
        st.warning(f"手続ID {proc_id} の詳細が見つかりません")
        return
    r = row.iloc[0]
    
    # タイトル部
    st.title(f"📄 {r.get('手続名', '')}")
    st.caption(f"手続ID: {r.get('手続ID','')} | 所管府省庁: {r.get('所管府省庁','')}")
    
    # 主要指標を上部に表示
    st.markdown("### 📊 主要指標")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("手続ID", r.get('手続ID', '—'))
    with col2:
        status = _normalize_label('オンライン化の実施状況', r.get('オンライン化の実施状況', ''))
        st.metric("オンライン化状況", status if status else "—")
    with col3:
        st.metric("総手続件数", f"{int(r.get('総手続件数', 0) or 0):,}")
    with col4:
        st.metric("オンライン手続件数", f"{int(r.get('オンライン手続件数', 0) or 0):,}")
    with col5:
        rate = float(r.get('オンライン化率', 0) or 0)
        st.metric("オンライン化率", f"{rate:.1f}%")
    
    st.divider()
    
    # 2カラムレイアウトで情報を整理
    col_left, col_right = st.columns(2)
    
    with col_left:
        # 基本情報
        with st.expander("🏛️ **基本情報**", expanded=True):
            items = [
                ("所管府省庁", r.get('所管府省庁', '—')),
                ("手続名", r.get('手続名', '—')),
                ("手続類型", _normalize_label('手続類型', r.get('手続類型', '—'))),
                ("手続主体", r.get('手続主体', '—')),
                ("手続の受け手", r.get('手続の受け手', '—')),
                ("経由機関", r.get('経由機関', '—')),
                ("事務区分", r.get('事務区分', '—')),
                ("府省共通手続", r.get('府省共通手続', '—')),
            ]
            for label, value in items:
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.markdown(f"**{label}:**")
                with col2:
                    st.text(value if value else '—')
        
        # 法令情報
        with st.expander("⚖️ **法令情報**", expanded=True):
            items = [
                ("法令名", r.get('法令名', '—')),
                ("法令番号", r.get('法令番号', '—')),
                ("根拠条項号", r.get('根拠条項号', '—')),
            ]
            for label, value in items:
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.markdown(f"**{label}:**")
                with col2:
                    st.text(value if value else '—')
        
        # システム情報
        with st.expander("💻 **システム情報**", expanded=True):
            items = [
                ("申請システム", r.get('情報システム(申請)', '—')),
                ("事務処理システム", r.get('情報システム(事務処理)', '—')),
                ("処理期間(オンライン)", r.get('処理期間(オンライン)', '—')),
                ("処理期間(非オンライン)", r.get('処理期間(非オンライン)', '—')),
            ]
            for label, value in items:
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.markdown(f"**{label}:**")
                with col2:
                    st.text(value if value else '—')
    
    with col_right:
        # 申請・書類情報
        with st.expander("📝 **申請・書類情報**", expanded=True):
            items = [
                ("本人確認手法", r.get('申請等における本人確認手法', '—')),
                ("提出先機関", r.get('申請を提出する機関', '—')),
            ]
            for label, value in items:
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.markdown(f"**{label}:**")
                with col2:
                    st.text(value if value else '—')
            
            # 長いテキストの項目
            if pd.notna(r.get('申請書等に記載させる情報')):
                st.markdown("**申請書記載情報:**")
                st.info(r.get('申請書等に記載させる情報', '—'))
            
            if pd.notna(r.get('申請時に添付させる書類')):
                st.markdown("**添付書類:**")
                st.info(r.get('申請時に添付させる書類', '—'))
        
        # 手数料情報
        with st.expander("💰 **手数料情報**", expanded=True):
            items = [
                ("納付有無", r.get('手数料等の納付有無', '—')),
                ("納付方法", r.get('手数料等の納付方法', '—')),
                ("優遇措置", r.get('手数料等のオンライン納付時の優遇措置', '—')),
            ]
            for label, value in items:
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.markdown(f"**{label}:**")
                with col2:
                    st.text(value if value else '—')
        
        # ライフイベント・士業
        with st.expander("🌟 **ライフイベント・士業**", expanded=True):
            if pd.notna(r.get('手続が行われるイベント(個人)')):
                st.markdown("**個人ライフイベント:**")
                st.info(r.get('手続が行われるイベント(個人)', '—'))
            
            if pd.notna(r.get('手続が行われるイベント(法人)')):
                st.markdown("**法人ライフイベント:**")
                st.info(r.get('手続が行われるイベント(法人)', '—'))
            
            if pd.notna(r.get('申請に関連する士業')):
                st.markdown("**関連士業:**")
                st.info(r.get('申請に関連する士業', '—'))
    
    # 全項目データ（折りたたみ）
    with st.expander("📋 **全38項目の詳細データ**", expanded=False):
        # 重要な項目を先頭に配置
        important_cols = ['手続ID', '手続名', '法令名', '所管府省庁', 'オンライン化の実施状況']
        other_cols = [c for c in COLUMNS if c not in important_cols]
        ordered_cols = important_cols + other_cols
        
        data_dict = {}
        for col in ordered_cols:
            if col in r:
                value = r[col]
                if pd.notna(value) and str(value).strip():
                    data_dict[col] = str(value)
                else:
                    data_dict[col] = '—'
        
        display_df = pd.DataFrame.from_dict(data_dict, orient='index', columns=['値'])
        display_df.index.name = '項目名'
        st.dataframe(display_df, use_container_width=True, height=400)
    
    # CSVエクスポート
    st.divider()
    csv_data = pd.DataFrame([r]).to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
    st.download_button(
        label="📥 この手続の情報をCSVでダウンロード",
        data=csv_data,
        file_name=f"procedure_{proc_id}.csv",
        mime="text/csv"
    )

def main():
    """メインアプリケーション"""
    
    # タイトル
    st.title("⚖️ 日本の法令に基づく行政手続等オンライン化状況ダッシュボード")
    st.markdown("約75,000件の法令・行政手続データを可視化・分析")
    
    # データ読み込み（初回のみ）
    if not st.session_state.data_loaded:
        st.session_state.df = load_data()
        st.session_state.data_loaded = True
    
    df = st.session_state.df

    # サイドバー
    with st.sidebar:
        st.header("📋 フィルター設定")

        # --- 即時適用フィルター ---
        st.markdown("**府省庁**")
        all_ministries = get_unique_values(df, '所管府省庁')
        # 建制順（歴史的な省庁設立順）に並べ替え
        # 明治期からの伝統的省庁 → 戦後設立 → 平成再編 → 近年設立の順
        ministry_order = [
            "宮内庁",           # 1869年（明治2年）宮内省として設立
            "法務省",           # 1871年（明治4年）司法省として設立
            "外務省",           # 1869年（明治2年）外務省設立
            "財務省",           # 1869年（明治2年）大蔵省として設立、2001年財務省に
            "文部科学省",       # 1871年（明治4年）文部省として設立、2001年文部科学省に
            "農林水産省",       # 1881年（明治14年）農商務省として設立
            "経済産業省",       # 1881年（明治14年）農商務省、1949年通商産業省、2001年経済産業省に
            "国土交通省",       # 1874年（明治7年）内務省、2001年国土交通省に
            "会計検査院",       # 1880年（明治13年）会計検査院設立
            "人事院",           # 1948年（昭和23年）人事院設立
            "内閣官房",         # 1947年（昭和22年）内閣官房設立
            "総務省",           # 1960年（昭和35年）自治省、2001年総務省に
            "厚生労働省",       # 1938年（昭和13年）厚生省、2001年厚生労働省に
            "防衛省",           # 1954年（昭和29年）防衛庁、2007年防衛省に
            "国家公安委員会",   # 1954年（昭和29年）国家公安委員会設立
            "公正取引委員会",   # 1947年（昭和22年）公正取引委員会設立
            "環境省",           # 1971年（昭和46年）環境庁、2001年環境省に
            "内閣府",           # 2001年（平成13年）内閣府設立
            "金融庁",           # 1998年（平成10年）金融監督庁、2000年金融庁に
            "消費者庁",         # 2009年（平成21年）消費者庁設立
            "復興庁",           # 2012年（平成24年）復興庁設立
            "個人情報保護委員会", # 2016年（平成28年）個人情報保護委員会設立
            "カジノ管理委員会", # 2020年（令和2年）カジノ管理委員会設立
            "デジタル庁"        # 2021年（令和3年）デジタル庁設立
        ]
        # 順序付きリストを作成（存在するものだけ）
        ordered_ministries = [m for m in ministry_order if m in all_ministries]
        # リストにない府省庁を追加
        remaining = [m for m in all_ministries if m not in ordered_ministries]
        ordered_ministries.extend(sorted(remaining))
        
        selected_ministries = st.multiselect(
            "府省庁を選択",
            ordered_ministries,
            key="ministry_filter",
            label_visibility="collapsed",
            help=FIELD_DEFS.get('所管府省庁', '')
        )

        st.markdown("**オンライン化状況**")
        all_statuses = get_unique_values(df, 'オンライン化の実施状況')
        selected_statuses = st.multiselect(
            "状況を選択",
            all_statuses,
            key="status_filter",
            label_visibility="collapsed",
            help=FIELD_DEFS.get('オンライン化の実施状況', '')
        )

        st.markdown("**手続類型**")
        all_types = get_unique_values(df, '手続類型')
        selected_types = st.multiselect(
            "類型を選択",
            all_types,
            key="type_filter",
            label_visibility="collapsed",
            help=FIELD_DEFS.get('手続類型', '')
        )

        st.markdown("**手続主体**")
        all_actors = get_unique_values(df, '手続主体') if '手続主体' in df.columns else []
        selected_actors = st.multiselect(
            "主体を選択",
            all_actors,
            key="actor_filter",
            label_visibility="collapsed",
            help=FIELD_DEFS.get('手続主体', '')
        )

        st.markdown("**手続の受け手**")
        all_receivers = get_unique_values(df, '手続の受け手') if '手続の受け手' in df.columns else []
        selected_receivers = st.multiselect(
            "受け手を選択",
            all_receivers,
            key="receiver_filter",
            label_visibility="collapsed",
            help=FIELD_DEFS.get('手続の受け手', '')
        )

        st.markdown("**事務区分**")
        all_office_types = get_unique_values(df, '事務区分') if '事務区分' in df.columns else []
        selected_office_types = st.multiselect(
            "事務区分を選択",
            all_office_types,
            key="office_type_filter",
            label_visibility="collapsed",
            help=FIELD_DEFS.get('事務区分', '')
        )

        st.markdown("**府省共通手続**")
        all_common = get_unique_values(df, '府省共通手続') if '府省共通手続' in df.columns else []
        selected_common = st.multiselect(
            "共通手続の種別を選択",
            all_common,
            key="common_filter",
            label_visibility="collapsed",
            help=FIELD_DEFS.get('府省共通手続', '')
        )

        st.markdown("**手続件数範囲**")
        count_ranges = [
            "100万件以上",
            "10万件以上100万件未満",
            "1万件以上10万件未満",
            "1000件以上1万件未満",
            "100件以上1000件未満",
            "10件以上100件未満",
            "1件以上10件未満",
            "0件もしくは不明"
        ]
        selected_count_ranges = st.multiselect(
            "手続件数範囲を選択",
            count_ranges,
            key="count_range_filter",
            label_visibility="collapsed",
            help="総手続件数による絞り込み"
        )

        # 即時フィルタリング実行
        filtered_df = filter_dataframe(
            df,
            selected_ministries,
            selected_statuses,
            selected_types,
            selected_receivers,
            actors=selected_actors,
            receivers=selected_receivers,
            office_types=selected_office_types,
            is_common=selected_common,
            count_ranges=selected_count_ranges,
        )

        with st.expander("ℹ️ 項目定義（抜粋）"):
            for k in ["手続類型", "手続主体", "手続の受け手", "事務区分", "府省共通手続", "オンライン化の実施状況"]:
                if k in FIELD_DEFS:
                    st.markdown(f"**{k}** — {FIELD_DEFS[k]}")
    
    # 詳細画面の表示（検索結果から遷移）
    if st.session_state.get('show_detail', False) and st.session_state.get('selected_procedure_id'):
        # 戻るボタン
        if st.button("← 検索結果に戻る"):
            st.session_state['show_detail'] = False
            st.session_state['selected_procedure_id'] = None
            st.rerun()
        
        # 詳細表示
        _render_procedure_detail(st.session_state['selected_procedure_id'], df)
        return
    
    # メインコンテンツ
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 概要統計", 
        "⚖️ 法令別分析", 
        "🏢 府省庁別分析",
        "💻 申請システム分析",
        "📎 申請文書分析",
        "🔍 法令・手続検索",
        "🤖 高度な分析(β)"
    ])
    
    with tab1:
        st.header("📊 概要統計")

        # KPIカード（カラムの存在を確認しつつ安全に算出）
        col1, col2, col3, col4 = st.columns(4)
        n_total = len(filtered_df)
        with col1:
            delta_val = n_total - len(df)
            st.metric("総手続数", f"{n_total:,}", delta=(f"{delta_val:+,}" if delta_val != 0 else None))
        with col2:
            total_proc_count = filtered_df['総手続件数'].sum() if '総手続件数' in filtered_df.columns else 0
            st.metric("総手続件数", f"{int(total_proc_count):,}")
        with col3:
            online_count = filtered_df['オンライン手続件数'].sum() if 'オンライン手続件数' in filtered_df.columns else 0
            st.metric("オンライン手続件数", f"{int(online_count):,}")
        with col4:
            online_rate = (online_count / total_proc_count * 100) if total_proc_count else 0
            st.metric("オンライン化率", f"{online_rate:.1f}%")

        # グラフ
        col1, col2 = st.columns(2)

        with col1:
            # オンライン化状況の円グラフ（正規化適用）
            status_counts = normalized_counts(filtered_df, 'オンライン化の実施状況', 'オンライン化の実施状況')
            if status_counts.sum() > 0:
                status_df = status_counts.reset_index()
                status_df.columns = ['オンライン化の実施状況', '件数']
                fig_pie = px.pie(
                    status_df,
                    values='件数',
                    names='オンライン化の実施状況',
                    title="オンライン化の実施状況",
                    hole=0.4
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("該当するデータがありません（円グラフ）")

        with col2:
            # 手続類型の棒グラフ（正規化適用）
            type_counts = normalized_counts(filtered_df, '手続類型', '手続類型')
            # 定義順があればhead(10)後でもOK、なければ頻度上位10
            if '手続類型' in OPTION_ORDERS:
                type_counts = type_counts.head(10)
            else:
                type_counts = type_counts.head(10)
            if type_counts.sum() > 0:
                type_df = type_counts.reset_index()
                type_df.columns = ['手続類型', '件数']
                # 降順にソート（グラフ上で上から下へ多い順に表示）
                type_df = type_df.sort_values('件数', ascending=True)
                fig_bar = px.bar(
                    type_df,
                    x='件数',
                    y='手続類型',
                    orientation='h',
                    title="手続類型TOP10",
                    labels={'件数': '件数', '手続類型': '手続類型'}
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.info("該当するデータがありません（棒グラフ）")
        
        # 手続一覧の表示
        st.subheader("📋 手続一覧")
        
        # 表示する列を選択
        display_columns = [
            '手続ID', '手続名', '所管府省庁', '手続類型',
            'オンライン化の実施状況', '総手続件数', 
            'オンライン手続件数', 'オンライン化率'
        ]
        
        # 存在する列のみ選択
        available_columns = [col for col in display_columns if col in filtered_df.columns]
        
        # 選択可能なデータフレームを表示
        selection = st.dataframe(
            filtered_df[available_columns].reset_index(drop=True),
            use_container_width=True,
            height=400,
            selection_mode="single-row",
            on_select="rerun"
        )
        
        # 選択された行がある場合、詳細をダイアログで表示
        if selection and selection.selection.rows:
            selected_idx = selection.selection.rows[0]
            selected_proc = filtered_df.iloc[selected_idx]
            
            @st.dialog(f"📄 手続詳細: {selected_proc['手続名'][:30]}...", width="large")
            def show_procedure_modal():
                _render_procedure_detail(selected_proc['手続ID'], df)
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.info(f"選択された手続: {selected_proc['手続名']}")
            with col2:
                if st.button("📄 詳細を表示", key="overview_detail_btn"):
                    show_procedure_modal()
        
        # CSVダウンロードボタン
        csv_data = df_to_csv_bytes(filtered_df[available_columns])
        st.download_button(
            label="📥 手続一覧をCSVダウンロード",
            data=csv_data,
            file_name="procedures_list.csv",
            mime="text/csv"
        )
    
    with tab2:
        st.header("⚖️ 法令別分析")
        
        # 法令別の手続数
        st.subheader("📚 法令別手続数（TOP20）")
        law_counts = filtered_df['法令名'].value_counts().head(20)
        if len(law_counts) > 0:
            # 降順にソート（少ない順から多い順へ、グラフ上で上から下へ多い順に表示）
            law_counts = law_counts.sort_values(ascending=True)
            fig_law = px.bar(
                x=law_counts.values,
                y=law_counts.index,
                orientation='h',
                title="法令別手続数",
                labels={'x': '手続数', 'y': '法令名'}
            )
            fig_law.update_layout(height=600)
            st.plotly_chart(fig_law, use_container_width=True)
        
        # 法令別のオンライン化状況
        st.subheader("📊 主要法令のオンライン化状況")
        
        # 手続数が多い法令TOP10のオンライン化状況
        top_laws = filtered_df['法令名'].value_counts().head(10).index
        law_online_data = []
        
        for law in top_laws:
            law_df = filtered_df[filtered_df['法令名'] == law]
            total = len(law_df)
            online = len(law_df[law_df['オンライン化の実施状況'].str.contains('実施済', na=False)])
            rate = (online / total * 100) if total > 0 else 0
            law_online_data.append({
                '法令名': law[:30] + ('...' if len(law) > 30 else ''),
                '手続数': total,
                'オンライン化済': online,
                'オンライン化率': rate
            })
        
        law_online_df = pd.DataFrame(law_online_data)
        
        fig_law_online = px.scatter(
            law_online_df,
            x='手続数',
            y='オンライン化率',
            size='手続数',
            hover_data=['法令名', 'オンライン化済'],
            title="主要法令のオンライン化率",
            labels={'オンライン化率': 'オンライン化率 (%)'}
        )
        st.plotly_chart(fig_law_online, use_container_width=True)
        
        # 法令別詳細テーブル
        st.subheader("📋 法令別詳細統計")
        st.dataframe(law_online_df.sort_values('オンライン化率', ascending=False), use_container_width=True)
        
        # 法令番号の形式別分析
        st.subheader("⚖️ 法令種別の分析")
        
        # 法律、政令、省令などの分類
        def classify_law_type(law_number):
            if pd.isna(law_number):
                return '不明'
            law_str = str(law_number)
            if '法律' in law_str:
                return '法律'
            elif '政令' in law_str:
                return '政令'
            elif '省令' in law_str or '規則' in law_str:
                return '省令・規則'
            elif '告示' in law_str:
                return '告示'
            elif '通達' in law_str or '通知' in law_str:
                return '通達・通知'
            else:
                return 'その他'
        
        filtered_df['法令種別'] = filtered_df['法令番号'].apply(classify_law_type)
        law_type_counts = filtered_df['法令種別'].value_counts()
        
        col1, col2 = st.columns(2)
        with col1:
            fig_law_type = px.pie(
                values=law_type_counts.values,
                names=law_type_counts.index,
                title="法令種別の分布",
                hole=0.4
            )
            st.plotly_chart(fig_law_type, use_container_width=True)
        
        with col2:
            # 法令種別ごとのオンライン化率
            law_type_online = filtered_df.groupby('法令種別').agg({
                '手続ID': 'count',
                'オンライン化率': 'mean'
            }).reset_index()
            law_type_online.columns = ['法令種別', '手続数', '平均オンライン化率']
            
            fig_law_type_online = px.bar(
                law_type_online,
                x='法令種別',
                y='平均オンライン化率',
                title="法令種別ごとの平均オンライン化率",
                labels={'平均オンライン化率': '平均オンライン化率 (%)'}
            )
            st.plotly_chart(fig_law_type_online, use_container_width=True)
    
    with tab3:
        st.header("🏢 府省庁別分析")
        
        # 府省庁別の手続数
        ministry_counts = filtered_df['所管府省庁'].value_counts().head(20)
        fig_ministry = px.bar(
            x=ministry_counts.index,
            y=ministry_counts.values,
            title="府省庁別手続数（TOP20）",
            labels={'x': '府省庁', 'y': '手続数'}
        )
        fig_ministry.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_ministry, use_container_width=True)
        
        # 府省庁別のオンライン化率
        ministry_stats = filtered_df.groupby('所管府省庁').agg({
            '手続ID': 'count',
            '総手続件数': 'sum',
            'オンライン手続件数': 'sum'
        }).reset_index()
        ministry_stats.columns = ['府省庁', '手続数', '総手続件数', 'オンライン手続件数']
        ministry_stats['オンライン化率'] = (
            ministry_stats['オンライン手続件数'] / ministry_stats['総手続件数'] * 100
        ).round(2)
        ministry_stats = ministry_stats[ministry_stats['総手続件数'] > 0]
        ministry_stats = ministry_stats.sort_values('オンライン化率', ascending=False).head(20)
        
        fig_ministry_online = px.scatter(
            ministry_stats,
            x='手続数',
            y='オンライン化率',
            size='総手続件数',
            hover_data=['府省庁', '総手続件数', 'オンライン手続件数'],
            title="府省庁別オンライン化率（バブルサイズ：総手続件数）",
            labels={'オンライン化率': 'オンライン化率 (%)'}
        )
        st.plotly_chart(fig_ministry_online, use_container_width=True)
        
        # 府省庁別詳細テーブル
        st.subheader("📋 府省庁別詳細統計")
        st.dataframe(
            ministry_stats.sort_values('オンライン化率', ascending=False),
            use_container_width=True
        )
    
    with tab4:
        st.header("💻 申請システム分析")
        
        # 申請システム（申請）の分析
        st.subheader("📊 申請システムの利用状況")
        
        # 申請システムのデータを集計
        system_df = filtered_df[filtered_df['情報システム(申請)'].notna()].copy()
        
        if len(system_df) > 0:
            # システム別の手続数を集計
            system_counts = system_df['情報システム(申請)'].value_counts().head(20)
            # 降順にソート（グラフ上で上から下へ多い順に表示）
            system_counts = system_counts.sort_values(ascending=True)
            
            # 申請システム別手続数の棒グラフ
            fig_system = px.bar(
                x=system_counts.values,
                y=system_counts.index,
                orientation='h',
                title="申請システム別手続数（TOP20）",
                labels={'x': '手続数', 'y': '申請システム'}
            )
            st.plotly_chart(fig_system, use_container_width=True)
            
            # システム別のオンライン化率
            system_stats = system_df.groupby('情報システム(申請)').agg({
                '手続ID': 'count',
                '総手続件数': 'sum',
                'オンライン手続件数': 'sum'
            }).reset_index()
            system_stats.columns = ['申請システム', '手続数', '総手続件数', 'オンライン手続件数']
            system_stats['オンライン化率'] = (
                system_stats['オンライン手続件数'] / system_stats['総手続件数'] * 100
            ).round(2)
            system_stats = system_stats[system_stats['総手続件数'] > 0]
            system_stats = system_stats.sort_values('オンライン化率', ascending=False).head(20)
            
            # 散布図：システム別オンライン化率
            fig_system_scatter = px.scatter(
                system_stats,
                x='手続数',
                y='オンライン化率',
                size='総手続件数',
                hover_data=['申請システム', '総手続件数', 'オンライン手続件数'],
                title="申請システム別オンライン化率（バブルサイズ：総手続件数）",
                labels={'オンライン化率': 'オンライン化率 (%)'}
            )
            st.plotly_chart(fig_system_scatter, use_container_width=True)
            
            # システム別詳細テーブル
            st.subheader("📋 申請システム別詳細統計")
            st.dataframe(
                system_stats[['申請システム', '手続数', '総手続件数', 'オンライン手続件数', 'オンライン化率']],
                use_container_width=True
            )
        else:
            st.info("申請システムのデータがありません")
        
        # 事務処理システムの分析
        st.subheader("🖥️ 事務処理システムの利用状況")
        
        # 事務処理システムのデータを集計
        process_system_df = filtered_df[filtered_df['情報システム(事務処理)'].notna()].copy()
        
        if len(process_system_df) > 0:
            # システム別の手続数を集計
            process_system_counts = process_system_df['情報システム(事務処理)'].value_counts().head(20)
            # 降順にソート（グラフ上で上から下へ多い順に表示）
            process_system_counts = process_system_counts.sort_values(ascending=True)
            
            # 事務処理システム別手続数の棒グラフ
            fig_process_system = px.bar(
                x=process_system_counts.values,
                y=process_system_counts.index,
                orientation='h',
                title="事務処理システム別手続数（TOP20）",
                labels={'x': '手続数', 'y': '事務処理システム'}
            )
            st.plotly_chart(fig_process_system, use_container_width=True)
            
            # 申請システムと事務処理システムの組み合わせ分析
            st.subheader("🔄 申請システムと事務処理システムの連携")
            
            # 両方のシステムを持つデータ
            both_systems = filtered_df[
                (filtered_df['情報システム(申請)'].notna()) & 
                (filtered_df['情報システム(事務処理)'].notna())
            ].copy()
            
            if len(both_systems) > 0:
                # システム組み合わせの集計
                system_combo = both_systems.groupby(['情報システム(申請)', '情報システム(事務処理)']).size().reset_index(name='手続数')
                system_combo = system_combo.sort_values('手続数', ascending=False).head(20)

                # ノード集合とインデックスを一度だけ構築
                nodes = list(pd.unique(pd.concat([
                    system_combo['情報システム(申請)'], system_combo['情報システム(事務処理)']
                ], ignore_index=True)))
                idx = {name: i for i, name in enumerate(nodes)}

                sources = [idx[s] for s in system_combo['情報システム(申請)']]
                targets = [idx[t] for t in system_combo['情報システム(事務処理)']]
                values  = system_combo['手続数'].tolist()

                # 可読性向上：ラベル改行・ノード間隔・太さ・図サイズを調整
                labels_raw = nodes
                labels_wrapped = [_wrap_label(n, width=10, max_lines=3) for n in labels_raw]

                sankey = go.Sankey(
                    arrangement='snap',
                    node=dict(
                        pad=24,
                        thickness=24,
                        line=dict(color="rgba(0,0,0,0.25)", width=0.5),
                        label=labels_wrapped,
                        hovertemplate="%{label}<extra></extra>",
                    ),
                    link=dict(
                        source=sources,
                        target=targets,
                        value=values,
                        color='rgba(150,150,150,0.25)',
                        hovertemplate="%{source.label} → %{target.label}<br>手続数: %{value}<extra></extra>",
                    )
                )

                fig_sankey = go.Figure(data=[sankey])
                fig_sankey.update_layout(
                    title="申請システムから事務処理システムへの連携（TOP20）",
                    font=dict(size=12),
                    hoverlabel=dict(font_size=12),
                    margin=dict(l=10, r=10, t=40, b=10),
                    height=800
                )
                st.plotly_chart(fig_sankey, use_container_width=True)

                st.caption("※ ノード名は10文字ごとに改行・最大3行で省略表示。ホバーで全名を確認できます。")

                st.dataframe(
                    system_combo,
                    use_container_width=True
                )
            else:
                st.info("両システムの連携データがありません")
        else:
            st.info("事務処理システムのデータがありません")

    # --- 添付書類分析タブ (tab5) ---
    with tab5:
        st.header("📎 申請文書分析")
        st.caption("添付書類や提出方式・電子署名の分布、手続類型との関係を分析します。")

        att_col = '申請時に添付させる書類'
        remove_col = '添付書類等提出の撤廃/省略状況'
        method_col = '添付書類等の提出方法'
        sign_col = '添付書類等への電子署名'

        cols_exist = [c for c in [att_col, remove_col, method_col, sign_col] if c in filtered_df.columns]
        if not cols_exist:
            st.info("添付書類に関する列が見つかりません")
        else:
            # --- 上段：サマリー（円グラフ×3） ---
            dist_cols = []
            if remove_col in filtered_df.columns:
                dist_cols.append((remove_col, '撤廃/省略状況の分布'))
            if method_col in filtered_df.columns:
                dist_cols.append((method_col, '提出方法の分布'))
            if sign_col in filtered_df.columns:
                dist_cols.append((sign_col, '電子署名の要否の分布'))

            if dist_cols:
                pie_top = st.slider("分布グラフのカテゴリ上限 (Top N)", 5, 15, 8, step=1, help="カテゴリが多い場合は上位のみ表示し、残りは『その他』にまとめます")
                for cname, title_txt in dist_cols:
                    series = filtered_df[cname].dropna().astype(str)
                    series = series[series.str.strip() != '']
                    if len(series) > 0:
                        dfv = _topn_with_other(series, top=pie_top, other_label='その他')
                        dfv[cname] = dfv['label'].map(lambda s: _wrap_label(s, width=10, max_lines=2))
                        fig = px.pie(dfv, values='件数', names=cname, title=title_txt, hole=0.4)
                        fig.update_layout(
                            margin=dict(l=0, r=0, t=40, b=0),
                            height=400,
                            legend=dict(font=dict(size=11), tracegroupgap=4)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info(f"'{cname}' のデータがありません")

            st.divider()

            # --- 中段：添付書類トップ ---
            st.subheader("📌 添付書類の頻出（TOP N）")
            if att_col in filtered_df.columns:
                att_series = filtered_df[att_col].dropna().apply(_split_multi_values).explode().astype(str)
                att_series = att_series[att_series.str.strip() != '']
                if len(att_series) > 0:
                    top_k = st.slider("表示件数（添付書類TOP）", 10, 50, 20, step=5)
                    att_counts = att_series.value_counts().head(top_k)
                    att_df = att_counts.reset_index()
                    att_df.columns = ['添付書類', '件数']
                    # 降順にソート（グラフ上で上から下へ多い順に表示）
                    att_df = att_df.sort_values('件数', ascending=True)
                    fig_att = px.bar(
                        att_df,
                        x='件数', y='添付書類', orientation='h',
                        title=f"添付書類の頻出（TOP{top_k})",
                        labels={'件数': '件数', '添付書類': '添付書類'}
                    )
                    fig_att.update_layout(margin=dict(l=0, r=0, t=40, b=0), height=520)
                    st.plotly_chart(fig_att, use_container_width=True)
                    with st.expander("📥 集計CSVをダウンロード"):
                        st.download_button("添付書類TOPのCSV", df_to_csv_bytes(att_df), file_name="attachment_top.csv", mime="text/csv")
                else:
                    st.info("添付書類の値が見つかりません")

            st.divider()

            # --- 下段：クロス集計（添付書類 × 手続類型） ---
            if att_col in filtered_df.columns and '手続類型' in filtered_df.columns:
                st.subheader("📊 添付書類 × 手続類型")
                att_series_for_ct = filtered_df[att_col].dropna().apply(_split_multi_values).explode().astype(str)
                att_series_for_ct = att_series_for_ct[att_series_for_ct.str.strip() != '']
                att_top = att_series_for_ct.value_counts().head(15).index if len(att_series_for_ct) > 0 else []
                if len(att_top) > 0:
                    exploded = filtered_df[[att_col, '手続類型']].dropna().copy()
                    exploded[att_col] = exploded[att_col].apply(_split_multi_values)
                    exploded = exploded.explode(att_col)
                    exploded = exploded[exploded[att_col].isin(att_top)]
                    ct = pd.crosstab(exploded[att_col], exploded['手続類型'])
                    if ct.sum().sum() > 0:
                        ct = ct.loc[(ct.sum(axis=1)).sort_values(ascending=False).index]
                        fig_ct = px.imshow(ct, text_auto=True, aspect='auto', title='添付書類 × 手続類型（件数）')
                        st.plotly_chart(fig_ct, use_container_width=True)
                        with st.expander("📥 クロス集計CSV"):
                            st.download_button("クロス集計CSV", df_to_csv_bytes(ct.reset_index()), file_name="attachment_by_type.csv", mime="text/csv")
                else:
                    st.info("添付書類のトップが得られなかったため、クロス集計を表示できません")
    
    with tab6:
        st.header("⚖️ 法令・手続検索")
        st.caption("法令名、法令番号、根拠条項、手続名、手続IDなどで検索できます")

        # --- 法令検索フォーム ---
        st.subheader("📚 法令による検索")
        col1, col2, col3 = st.columns(3)
        with col1:
            search_law_name = st.text_input(
                "法令名",
                placeholder="例：行政手続法",
                help=FIELD_DEFS.get('法令名', '')
            )
        with col2:
            search_law_number = st.text_input(
                "法令番号",
                placeholder="例：平成5年法律第88号",
                help=FIELD_DEFS.get('法令番号', '')
            )
        with col3:
            search_clause = st.text_input(
                "根拠条項号",
                placeholder="例：第3条",
                help=FIELD_DEFS.get('根拠条項号', '')
            )
        
        # --- 手続検索フォーム ---
        st.subheader("🔍 手続による検索")
        col1, col2 = st.columns(2)
        with col1:
            search_keyword = st.text_input(
                "手続名で検索",
                placeholder="例：許可申請",
                help=FIELD_DEFS.get('手続名', '')
            )
        with col2:
            search_id = st.text_input(
                "手続IDで検索",
                placeholder="例：12345",
                help="手続の固有ID"
            )

        # ライフイベント検索
        col1, col2 = st.columns(2)
        with col1:
            life_events_personal = st.multiselect(
                "ライフイベント（個人）",
                ["出生", "結婚", "引越し", "就職・転職", "介護", "死亡・相続", "その他"],
                help=FIELD_DEFS.get('手続が行われるイベント(個人)', '')
            )
        with col2:
            life_events_corporate = st.multiselect(
                "ライフイベント（法人）",
                ["法人の設立", "職員の採用・退職", "事務所の新設・移転", "法人の合併・分割", "その他"],
                help=FIELD_DEFS.get('手続が行われるイベント(法人)', '')
            )

        # 士業検索
        professions = st.multiselect(
            "関連する士業",
            ["弁護士", "司法書士", "行政書士", "税理士", "社会保険労務士", "公認会計士", "弁理士"],
            help=FIELD_DEFS.get('申請に関連する士業', '')
        )

        # --- 検索実行 ---
        search_df = filtered_df.copy()

        # 法令による絞り込み
        if search_law_name:
            search_df = search_df[search_df['法令名'].str.contains(search_law_name, na=False)]
        
        if search_law_number:
            search_df = search_df[search_df['法令番号'].str.contains(search_law_number, na=False)]
        
        if search_clause:
            search_df = search_df[search_df['根拠条項号'].str.contains(search_clause, na=False)]

        # 手続による絞り込み
        if search_keyword:
            search_df = search_df[search_df['手続名'].str.contains(search_keyword, na=False)]

        if search_id:
            search_df = search_df[search_df['手続ID'] == search_id]

        if life_events_personal:
            mask = search_df['手続が行われるイベント(個人)'].apply(
                lambda x: any(event in str(x) for event in life_events_personal)
            )
            search_df = search_df[mask]

        if life_events_corporate:
            mask = search_df['手続が行われるイベント(法人)'].apply(
                lambda x: any(event in str(x) for event in life_events_corporate)
            )
            search_df = search_df[mask]

        if professions:
            mask = search_df['申請に関連する士業'].apply(
                lambda x: any(prof in str(x) for prof in professions)
            )
            search_df = search_df[mask]

        # --- 結果表示＆ダウンロード ---
        st.subheader(f"検索結果: {len(search_df)}件")

        if len(search_df) > 0:
            # 表示カラムの選択（法令情報を重視）
            available_columns = list(search_df.columns)
            default_columns = ["手続ID", "手続名", "法令名", "法令番号", "根拠条項号", "所管府省庁", "オンライン化の実施状況"]
            # 存在するカラムのみを含める
            default_columns = [col for col in default_columns if col in available_columns]

            display_columns = st.multiselect(
                "表示する項目を選択",
                available_columns,
                default=default_columns
            )

            if display_columns:
                # データフレーム表示用のコピーを作成
                display_df = search_df[display_columns].head(100).copy()
                
                # StreamlitのColumn Configurationを使用してクリック可能にする
                if '手続ID' in display_columns:
                    # 手続IDがある場合は、選択機能を追加
                    st.write("**📋 検索結果** ※行を選択して詳細ボタンをクリック")
                    
                    # データエディタで選択可能にする
                    event = st.dataframe(
                        display_df,
                        use_container_width=True,
                        height=400,
                        on_select="rerun",
                        selection_mode="single-row",
                        key="search_result_table"
                    )
                    
                    # 選択された行がある場合
                    if event.selection and event.selection.rows:
                        selected_row_idx = event.selection.rows[0]
                        selected_procedure = display_df.iloc[selected_row_idx]
                        
                        @st.dialog(f"📄 手続詳細: {selected_procedure.get('手続名', '')[:30]}...", width="large")
                        def show_search_modal():
                            _render_procedure_detail(selected_procedure['手続ID'], df)
                        
                        col1, col2, col3 = st.columns([2, 1, 4])
                        with col1:
                            st.info(f"選択: {selected_procedure['手続ID']}")
                        with col2:
                            if st.button("詳細を表示", type="primary"):
                                show_search_modal()
                        with col3:
                            # 手続名を表示
                            st.text(f"{selected_procedure.get('手続名', '')[:40]}...")
                else:
                    # 手続IDがない場合は通常のデータフレーム表示
                    st.dataframe(
                        display_df,
                        use_container_width=True,
                        height=400
                    )

                # CSVダウンロード（キャッシュ利用）
                st.download_button(
                    label="📥 検索結果をCSVでダウンロード",
                    data=df_to_csv_bytes(search_df[display_columns]),
                    file_name="search_results.csv",
                    mime="text/csv"
                )
    
    
    with tab7:
        st.header("🤖 高度な分析(β)")
        st.info("ベータ版：機械学習を用いた高度な分析機能を提供しています")
        
        # サブタブを作成
        analysis_tab1, analysis_tab2, analysis_tab3 = st.tabs([
            "🔬 クラスタリング・優先度分析",
            "📊 相関分析",
            "🕸️ ネットワーク分析"
        ])
        
        with analysis_tab1:
            # クラスタリング分析
            st.subheader("🔬 府省庁のクラスタリング分析")
            
            # 府省庁ごとの特徴量を計算
            ministry_features = filtered_df.groupby('所管府省庁').agg({
                '手続ID': 'count',
                '総手続件数': 'sum',
                'オンライン手続件数': 'sum',
                '非オンライン手続件数': 'sum'
            }).reset_index()
        
            ministry_features.columns = ['府省庁', '手続数', '総手続件数', 'オンライン手続件数', '非オンライン手続件数']
            ministry_features['オンライン化率'] = (
                ministry_features['オンライン手続件数'] / ministry_features['総手続件数'] * 100
            ).fillna(0)
            ministry_features['平均手続件数'] = ministry_features['総手続件数'] / ministry_features['手続数']
            
            # 十分なデータがある府省庁のみ対象
            ministry_features = ministry_features[ministry_features['総手続件数'] > 100]
            
            if len(ministry_features) > 3:
                # 特徴量の標準化
                features_for_clustering = ['手続数', 'オンライン化率', '平均手続件数']
                X = ministry_features[features_for_clustering]
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # K-meansクラスタリング
                n_clusters = min(4, len(ministry_features) - 1)
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                ministry_features['クラスター'] = kmeans.fit_predict(X_scaled)
                
                # PCAで2次元に削減して可視化
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X_scaled)
                ministry_features['PC1'] = X_pca[:, 0]
                ministry_features['PC2'] = X_pca[:, 1]
                
                # クラスタリング結果の可視化
                fig_cluster = px.scatter(
                    ministry_features,
                    x='PC1',
                    y='PC2',
                    color='クラスター',
                    hover_data=['府省庁', '手続数', 'オンライン化率', '平均手続件数'],
                    title=f"府省庁のクラスタリング結果（{n_clusters}クラスター）",
                    labels={'PC1': f'第1主成分 ({pca.explained_variance_ratio_[0]:.1%})',
                            'PC2': f'第2主成分 ({pca.explained_variance_ratio_[1]:.1%})',
                            'クラスター': 'クラスター'}
                )
                st.plotly_chart(fig_cluster, use_container_width=True)
                
                # クラスター別の特徴
                st.subheader("📊 クラスター別特徴")
                cluster_stats = ministry_features.groupby('クラスター')[features_for_clustering].mean().round(2)
                st.dataframe(cluster_stats, use_container_width=True)
                
                # クラスター別府省庁リスト
                st.subheader("📋 クラスター別府省庁")
                for cluster_id in sorted(ministry_features['クラスター'].unique()):
                    cluster_ministries = ministry_features[ministry_features['クラスター'] == cluster_id]['府省庁'].tolist()
                    st.write(f"**クラスター {cluster_id}:** {', '.join(cluster_ministries[:10])}{'...' if len(cluster_ministries) > 10 else ''}")
        
        # 時系列分析（オンライン化実施時期）
        st.subheader("📅 オンライン化実施時期の分析")
        
        # オンライン化実施時期の分布
        time_data = filtered_df[filtered_df['オンライン化実施時期'].notna()]['オンライン化実施時期'].value_counts()
        if len(time_data) > 0:
            fig_timeline = px.bar(
                x=time_data.index[:20],
                y=time_data.values[:20],
                title="オンライン化実施時期の分布（TOP20）",
                labels={'x': '実施時期', 'y': '手続数'}
            )
            fig_timeline.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_timeline, use_container_width=True)
            
            # 予測分析
            st.subheader("🔮 オンライン化推進の優先度分析")
            
            # オンライン化されていない手続で、手続件数が多いものを抽出
            not_online = filtered_df[
                (filtered_df['オンライン化の実施状況'].str.contains('未実施|検討中|予定なし', na=False)) &
                (filtered_df['総手続件数'] > 0)
            ].copy()
            
            if len(not_online) > 0:
                # 優先度スコアを計算（手続件数ベース）
                not_online['優先度スコア'] = not_online['総手続件数']
                
                # 府省庁ごとに優先度スコアを集計
                priority_by_ministry = not_online.groupby('所管府省庁')['優先度スコア'].sum().sort_values(ascending=False).head(15)
                
                fig_priority = px.bar(
                    x=priority_by_ministry.values,
                    y=priority_by_ministry.index,
                    orientation='h',
                    title="オンライン化優先度が高い府省庁（未実施手続の総件数ベース）",
                    labels={'x': '優先度スコア（総手続件数）', 'y': '府省庁'}
                )
                st.plotly_chart(fig_priority, use_container_width=True)
                
                # 優先度の高い個別手続
                st.subheader("🎯 オンライン化優先度TOP20手続")
                top_priority = not_online.nlargest(20, '優先度スコア')[[
                    '手続名', '所管府省庁', '総手続件数', 'オンライン化の実施状況'
                ]]
                st.dataframe(top_priority, use_container_width=True)
        
        with analysis_tab2:
            # 数値データの相関分析
            st.subheader("🔗 手続件数の相関関係")
            
            numeric_cols = ['総手続件数', 'オンライン手続件数', '非オンライン手続件数', 'オンライン化率']
            correlation_data = filtered_df[numeric_cols].corr()
        
            # ヒートマップの作成
            fig_heatmap = go.Figure(data=go.Heatmap(
            z=correlation_data.values,
            x=correlation_data.columns,
            y=correlation_data.columns,
            colorscale='RdBu',
            zmid=0,
            text=correlation_data.values.round(2),
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title="相関係数")
            ))
            fig_heatmap.update_layout(
                title="手続件数関連指標の相関行列",
                width=600,
                height=500
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # カテゴリカルデータの関連性分析
            st.subheader("📊 カテゴリ間の関連性")
        
            # オンライン化状況と手続類型のクロス集計
            cross_tab = pd.crosstab(
            filtered_df['手続類型'],
            filtered_df['オンライン化の実施状況'],
            normalize='index'
            ) * 100
            
            # 上位10の手続類型のみ表示
            top_types = filtered_df['手続類型'].value_counts().head(10).index
            cross_tab_top = cross_tab.loc[cross_tab.index.isin(top_types)]
            
            # スタックドバーチャート
            fig_stacked = go.Figure()
            for col in cross_tab_top.columns:
                fig_stacked.add_trace(go.Bar(
                    name=col,
                    y=cross_tab_top.index,
                    x=cross_tab_top[col],
                    orientation='h'
                ))
            
            fig_stacked.update_layout(
                title="手続類型別のオンライン化状況（％）",
                barmode='stack',
                xaxis_title="割合（％）",
                yaxis_title="手続類型",
                height=500
            )
            st.plotly_chart(fig_stacked, use_container_width=True)
            
            # 府省庁間の類似度分析
            st.subheader("🏢 府省庁間の類似度分析")
        
            # 府省庁ごとの手続類型の分布を計算
            ministry_procedure_matrix = pd.crosstab(
            filtered_df['所管府省庁'],
            filtered_df['手続類型']
            )
            
            # 主要な府省庁のみ選択（手続数が多い上位20）
            top_ministries = filtered_df['所管府省庁'].value_counts().head(20).index
            ministry_procedure_matrix = ministry_procedure_matrix.loc[
                ministry_procedure_matrix.index.isin(top_ministries)
            ]
            
            if len(ministry_procedure_matrix) > 2:
                # コサイン類似度を計算
                from sklearn.metrics.pairwise import cosine_similarity
                similarity_matrix = cosine_similarity(ministry_procedure_matrix)
                similarity_df = pd.DataFrame(
                    similarity_matrix,
                    index=ministry_procedure_matrix.index,
                    columns=ministry_procedure_matrix.index
                )
                
                # 類似度ヒートマップ
                fig_similarity = go.Figure(data=go.Heatmap(
                z=similarity_df.values,
                x=similarity_df.columns,
                y=similarity_df.index,
                colorscale='Viridis',
                text=similarity_df.values.round(2),
                texttemplate='%{text}',
                textfont={"size": 8},
                colorbar=dict(title="類似度")
                ))
                fig_similarity.update_layout(
                    title="府省庁間の手続類型類似度",
                    width=800,
                    height=700
                )
                fig_similarity.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_similarity, use_container_width=True)
                
                # 最も類似している府省庁ペア
                st.subheader("🤝 最も類似している府省庁ペアTOP10")
                similarity_pairs = []
                for i in range(len(similarity_df)):
                    for j in range(i+1, len(similarity_df)):
                        similarity_pairs.append({
                            '府省庁1': similarity_df.index[i],
                            '府省庁2': similarity_df.index[j],
                            '類似度': similarity_df.iloc[i, j]
                        })
                
                similarity_pairs_df = pd.DataFrame(similarity_pairs)
                top_similar = similarity_pairs_df.nlargest(10, '類似度')
                st.dataframe(top_similar, use_container_width=True)
        
        with analysis_tab3:
            st.subheader("🕸️ ネットワーク分析")

        # 共通の描画オプション
        col_opt1, col_opt2, col_opt3 = st.columns(3)
        with col_opt1:
            top_n = st.slider("対象ノードの上限 (TOP N)", 10, 100, 30, step=5, help="頻度の高いノードから上位N件に絞ります")
        with col_opt2:
            min_w = st.slider("エッジの最小重み (しきい値)", 1, 10, 2, step=1, help="共起回数や連携回数がこの値未満のエッジは非表示")
        with col_opt3:
            size_metric = st.selectbox("ノードサイズ指標", ["degree", "betweenness", "eigenvector", "pagerank"], index=0)

        # ネットワーク図の種類を選択
        network_type = st.radio(
            "ネットワーク図の種類を選択",
            ["府省庁間の連携ネットワーク", "手続類型の共起ネットワーク", "ライフイベントネットワーク"]
        )

        if network_type == "府省庁間の連携ネットワーク":
            st.subheader("🏢 府省庁間の連携ネットワーク")

            required_cols = ['所管府省庁', '府省共通手続']
            if not _has_cols(filtered_df, required_cols):
                st.stop()

            with st.spinner("ネットワークを構築中..."):
                common_procedures = filtered_df[_safe_notna(filtered_df['府省共通手続'])]
                if len(common_procedures) == 0:
                    st.warning("府省共通手続のデータがありません")
                else:
                    # 上位N府省庁に限定してノイズを抑制
                    ministries = filtered_df['所管府省庁'].value_counts().head(top_n)

                    # 府省庁ごとに「共通手続」の集合を作成
                    group_sets = (
                        common_procedures.groupby('所管府省庁')['府省共通手続']
                        .apply(lambda s: set([x for x in s.dropna().astype(str).tolist()]))
                        .to_dict()
                    )

                    G = nx.Graph()
                    for ministry in ministries.index:
                        G.add_node(ministry)

                    # 各ペアの共通数と正規化重み（Jaccard風）を計算
                    pairs = list(ministries.index)
                    for i in range(len(pairs)):
                        for j in range(i+1, len(pairs)):
                            a, b = pairs[i], pairs[j]
                            A, B = group_sets.get(a, set()), group_sets.get(b, set())
                            inter = len(A & B)
                            union = len(A | B)
                            if inter == 0 or union == 0:
                                continue
                            # 重み: 共通件数と正規化（Jaccard）を併記
                            w_raw = inter
                            w_norm = inter / union
                            if w_raw >= min_w:
                                G.add_edge(a, b, weight=w_raw, norm_weight=round(w_norm, 3))

                    # ノードサイズ = 選択した中心性
                    cent = _compute_centrality(G, size_metric)
                    sizes = _scale_sizes(cent, min_size=8, max_size=40)
                    for n in G.nodes():
                        G.nodes[n]['size'] = sizes.get(n, 12)
                        G.nodes[n]['tooltip'] = f"{n}<br>{size_metric}: {cent.get(n, 0):.3f}"

                    if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
                        st.warning("連携エッジを構成できませんでした（データ不足またはフィルタが厳しすぎます）")
                    else:
                        enable_interactive = st.toggle("ドラッグ可能なインタラクティブ表示（β）", value=True, key="net_ministry_interactive")
                        if enable_interactive:
                            _render_pyvis(G, height=700)
                        else:
                            pos = _layout_for_graph(G)

                            edge_trace = []
                            for (u, v, d) in G.edges(data=True):
                                x0, y0 = pos[u]
                                x1, y1 = pos[v]
                                w = d.get('weight', 1)
                                edge_trace.append(go.Scatter(
                                    x=[x0, x1, None], y=[y0, y1, None], mode='lines',
                                    line=dict(width=0.5 + w/5, color='#888'), hoverinfo='none'))

                            node_x, node_y, node_text, node_size = [], [], [], []
                            for node in G.nodes():
                                x, y = pos[node]
                                node_x.append(x); node_y.append(y)
                                node_text.append(G.nodes[node].get('tooltip', node))
                                node_size.append(G.nodes[node].get('size', 12))

                            node_trace = go.Scatter(
                                x=node_x, y=node_y, mode='markers+text', text=[str(n)[:10] for n in G.nodes()],
                                textposition="top center", hovertext=node_text, hoverinfo='text',
                                marker=dict(size=node_size, color='#1f77b4', line=dict(width=2, color='white'))
                            )

                            fig = go.Figure(data=edge_trace + [node_trace],
                                layout=go.Layout(title='府省庁間の連携ネットワーク', showlegend=False, hovermode='closest',
                                                 margin=dict(b=0,l=0,r=0,t=40),
                                                 xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                                 yaxis=dict(showgrid=False, zeroline=False, showticklabels=False), height=600))
                            st.plotly_chart(fig, use_container_width=True)
                        # CSVエクスポート
                        nodes_df, edges_df = _export_nodes_edges(G)
                        with st.expander("📥 ネットワークのCSVエクスポート"):
                            c1, c2 = st.columns(2)
                            with c1:
                                st.download_button("ノードCSVをダウンロード", df_to_csv_bytes(nodes_df), file_name="network_nodes.csv", mime="text/csv")
                            with c2:
                                st.download_button("エッジCSVをダウンロード", df_to_csv_bytes(edges_df), file_name="network_edges.csv", mime="text/csv")
                        st.info(f"ノード数: {G.number_of_nodes()}, エッジ数: {G.number_of_edges()}")
                        st.caption("※ エッジ重み=共通手続の件数（表示）、norm_weight=Jaccard風の正規化（内部属性）")

        elif network_type == "手続類型の共起ネットワーク":
            st.subheader("📝 手続類型の共起ネットワーク")
            if not _has_cols(filtered_df, ['所管府省庁', '手続類型']):
                st.stop()
            with st.spinner("共起ネットワークを構築中..."):
                # 出現頻度の高い手続類型に限定
                procedure_types = filtered_df['手続類型'].value_counts().head(top_n)
                target_types = set(procedure_types.index)

                # 府省庁ごとの手続類型集合
                ministry_groups = filtered_df.groupby('所管府省庁')['手続類型'].apply(lambda s: set([x for x in s.dropna().tolist()]))

                # 各タイプの出現数
                type_freq = {t: 0 for t in target_types}
                for types in ministry_groups:
                    for t in types:
                        if t in type_freq:
                            type_freq[t] += 1

                # 共起回数と正規化重み（cosine）
                from collections import defaultdict
                co_counts = defaultdict(int)
                for types in ministry_groups:
                    ts = [t for t in types if t in target_types]
                    for i in range(len(ts)):
                        for j in range(i+1, len(ts)):
                            a, b = sorted((ts[i], ts[j]))
                            co_counts[(a, b)] += 1

                G = nx.Graph()
                for t in target_types:
                    G.add_node(t)

                for (a, b), c in co_counts.items():
                    w = _cosine_normalized_weight(c, type_freq.get(a, 1), type_freq.get(b, 1))
                    if c >= min_w and w > 0:
                        G.add_edge(a, b, weight=c, norm_weight=round(w, 3))

                # ノードサイズ = 選択した中心性
                cent = _compute_centrality(G, size_metric)
                sizes = _scale_sizes(cent, min_size=8, max_size=40)
                for n in G.nodes():
                    G.nodes[n]['size'] = sizes.get(n, 12)
                    G.nodes[n]['tooltip'] = f"{n}<br>{size_metric}: {cent.get(n, 0):.3f}"

                if G.number_of_nodes() > 0:
                    enable_interactive = st.toggle("ドラッグ可能なインタラクティブ表示（β）", value=True, key="net_cooccurrence_interactive")
                    if enable_interactive:
                        _render_pyvis(G, height=700)
                    else:
                        pos = _layout_for_graph(G)

                        # エッジの描画
                        edge_trace = []
                        for edge in G.edges(data=True):
                            x0, y0 = pos[edge[0]]
                            x1, y1 = pos[edge[1]]
                            weight = edge[2].get('weight', 1)
                            edge_trace.append(go.Scatter(
                                x=[x0, x1, None],
                                y=[y0, y1, None],
                                mode='lines',
                                line=dict(width=0.5 + weight/5, color='#888'),
                                hoverinfo='none'
                            ))

                        # ノードの描画
                        node_x, node_y, node_text, node_size = [], [], [], []
                        for node in G.nodes():
                            x, y = pos[node]
                            node_x.append(x)
                            node_y.append(y)
                            node_text.append(G.nodes[node].get('tooltip', node))
                            node_size.append(G.nodes[node].get('size', 12))

                        node_trace = go.Scatter(
                            x=node_x,
                            y=node_y,
                            mode='markers',
                            hovertext=node_text,
                            hoverinfo='text',
                            marker=dict(
                                size=node_size,
                                color='#ff7f0e',
                                line=dict(width=2, color='white')
                            )
                        )

                        fig = go.Figure(data=edge_trace + [node_trace],
                                       layout=go.Layout(
                                           title='手続類型の共起ネットワーク',
                                           showlegend=False,
                                           hovermode='closest',
                                           margin=dict(b=0,l=0,r=0,t=40),
                                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                           height=600
                                       ))

                        st.plotly_chart(fig, use_container_width=True)
                    # CSVエクスポート
                    nodes_df, edges_df = _export_nodes_edges(G)
                    with st.expander("📥 ネットワークのCSVエクスポート"):
                        c1, c2 = st.columns(2)
                        with c1:
                            st.download_button("ノードCSVをダウンロード", df_to_csv_bytes(nodes_df), file_name="network_nodes.csv", mime="text/csv")
                        with c2:
                            st.download_button("エッジCSVをダウンロード", df_to_csv_bytes(edges_df), file_name="network_edges.csv", mime="text/csv")
                    st.info(f"ノード数: {G.number_of_nodes()}, エッジ数: {G.number_of_edges()}")
                    st.caption("※ エッジ重み=共起回数、norm_weight=cosine正規化（内部属性）")

        else:  # ライフイベントネットワーク
            st.subheader("🌟 ライフイベントネットワーク")
            if not _has_cols(filtered_df, ['手続類型', '手続が行われるイベント(個人)', '手続が行われるイベント(法人)']):
                st.stop()
            with st.spinner("ライフイベントネットワークを構築中..."):
                G = nx.Graph()

                def _split_events(val: Any) -> List[str]:
                    if val is None or (isinstance(val, float) and np.isnan(val)):
                        return []
                    s = str(val).strip()
                    if s.lower() == 'nan' or s == '':
                        return []
                    for sep in ['、', ',', '，', ';', '；']:
                        s = s.replace(sep, '、')
                    return [e.strip() for e in s.split('、') if e.strip()]

                from collections import Counter
                life_events_personal = []
                for events in filtered_df['手続が行われるイベント(個人)']:
                    life_events_personal.extend(_split_events(events))
                life_events_corporate = []
                for events in filtered_df['手続が行われるイベント(法人)']:
                    life_events_corporate.extend(_split_events(events))

                personal_counter = Counter(life_events_personal)
                corporate_counter = Counter(life_events_corporate)

                top_personal = dict(personal_counter.most_common(top_n//2))
                top_corporate = dict(corporate_counter.most_common(top_n//2))

                for event, count in top_personal.items():
                    G.add_node(f"個人: {event}", size=int(np.log(count + 1) * 10), category="personal")
                for event, count in top_corporate.items():
                    G.add_node(f"法人: {event}", size=int(np.log(count + 1) * 10), category="corporate")

                top_proc_types = set(filtered_df['手続類型'].value_counts().head(15).index)
                sub = filtered_df[filtered_df['手続類型'].isin(top_proc_types)]
                for _, row in sub.iterrows():
                    proc_type = row['手続類型']
                    for event in _split_events(row['手続が行われるイベント(個人)']):
                        node_name = f"個人: {event}"
                        if event in top_personal and node_name in G.nodes:
                            if proc_type not in G.nodes:
                                G.add_node(proc_type, size=5, category="procedure")
                            G.add_edge(node_name, proc_type)
                    for event in _split_events(row['手続が行われるイベント(法人)']):
                        node_name = f"法人: {event}"
                        if event in top_corporate and node_name in G.nodes:
                            if proc_type not in G.nodes:
                                G.add_node(proc_type, size=5, category="procedure")
                            G.add_edge(node_name, proc_type)

                # ノードサイズ = 選択した中心性（種類に関係なく）
                cent = _compute_centrality(G, size_metric)
                sizes = _scale_sizes(cent, min_size=8, max_size=40)
                for n in G.nodes():
                    G.nodes[n]['size'] = sizes.get(n, 12)
                    tt = f"{n}<br>{size_metric}: {cent.get(n, 0):.3f}"
                    G.nodes[n]['tooltip'] = tt

                if G.number_of_nodes() == 0:
                    st.warning("ネットワークを構成できませんでした（データ不足またはフィルタが厳しすぎます）")
                else:
                    enable_interactive = st.toggle("ドラッグ可能なインタラクティブ表示（β）", value=True, key="net_lifeevent_interactive")
                    if enable_interactive:
                        _render_pyvis(G, height=700)
                    else:
                        pos = _layout_for_graph(G)

                        edge_trace = []
                        for (u, v) in G.edges():
                            x0, y0 = pos[u]
                            x1, y1 = pos[v]
                            edge_trace.append(go.Scatter(
                                x=[x0, x1, None], y=[y0, y1, None], mode='lines',
                                line=dict(width=0.5, color='#888'), hoverinfo='none'))

                        node_traces = []
                        categories = {'personal': '#2ca02c', 'corporate': '#d62728', 'procedure': '#9467bd'}
                        symbols = {'personal': 'circle', 'corporate': 'square', 'procedure': 'diamond'}

                        for cat, color in categories.items():
                            node_x, node_y, node_text, node_size = [], [], [], []
                            for node in G.nodes():
                                if G.nodes[node].get('category', 'procedure') == cat:
                                    x, y = pos[node]
                                    node_x.append(x); node_y.append(y)
                                    node_text.append(G.nodes[node].get('tooltip', node))
                                    node_size.append(G.nodes[node].get('size', 10))
                            if node_x:
                                node_traces.append(go.Scatter(
                                    x=node_x, y=node_y, mode='markers', name={'personal': '個人イベント', 'corporate': '法人イベント', 'procedure': '手続類型'}[cat],
                                    hovertext=node_text, hoverinfo='text',
                                    marker=dict(size=node_size, color=color, symbol=symbols[cat], line=dict(width=2, color='white'))
                                ))

                        fig = go.Figure(data=edge_trace + node_traces,
                            layout=go.Layout(title='ライフイベントネットワーク', showlegend=True, hovermode='closest',
                                             margin=dict(b=0,l=0,r=0,t=40),
                                             xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                             yaxis=dict(showgrid=False, zeroline=False, showticklabels=False), height=600))
                        st.plotly_chart(fig, use_container_width=True)
                    # CSVエクスポート
                    nodes_df, edges_df = _export_nodes_edges(G)
                    with st.expander("📥 ネットワークのCSVエクスポート"):
                        c1, c2 = st.columns(2)
                        with c1:
                            st.download_button("ノードCSVをダウンロード", df_to_csv_bytes(nodes_df), file_name="network_nodes.csv", mime="text/csv")
                        with c2:
                            st.download_button("エッジCSVをダウンロード", df_to_csv_bytes(edges_df), file_name="network_edges.csv", mime="text/csv")
                    st.info(f"ノード数: {G.number_of_nodes()}, エッジ数: {G.number_of_edges()}")
                    st.caption("※ ノードサイズは中心性に基づきます（選択可能）")

if __name__ == "__main__":
    main()
