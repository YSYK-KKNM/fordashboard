
# Streamlit dashboard auto-generated to execute `indipro.ipynb` and render REAL charts
import streamlit as st
import json, io, contextlib, traceback
from pathlib import Path

st.set_page_config(page_title="Indipro – 实时图表", page_icon="📈", layout="wide")

st.title("Indipro 仪表盘（根据 Notebook 自动生成）")
st.caption("自动执行 indipro.ipynb 的代码并捕获图表/表格输出")

# ---- Controls ----
st.sidebar.header("运行设置")
run_btn = st.sidebar.button("▶ 运行 Notebook", type="primary")
show_code = st.sidebar.checkbox("显示每个单元格的代码", value=False)
stop_on_error = st.sidebar.checkbox("遇到错误时停止", value=False)

nb_path = Path("indipro.ipynb")
if not nb_path.exists():
    st.error("找不到 indipro.ipynb，请把该文件放在仓库根目录。")
    st.stop()

nb = json.loads(nb_path.read_text(encoding="utf-8"))

# Utility: execute code cell and collect matplotlib figures & pandas DataFrames
def execute_cells(cells):
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    out_items = []  # list of dicts: {'idx': i, 'code': code, 'figs': [figs], 'dfs': [(name, df)], 'stdout': text}
    ns = {}
    for i, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        code = "".join(cell.get("source", []))
        if not code.strip():
            continue
        # capture stdout
        stdout = io.StringIO()
        figs = []
        dfs = []
        try:
            with contextlib.redirect_stdout(stdout):
                # clear previous figures to capture only this cell's plots
                plt.close("all")
                exec(code, ns)
                # collect figures created in this cell
                fignums = plt.get_fignums()
                for num in fignums:
                    fig = plt.figure(num)
                    figs.append(fig)
                # collect DataFrames (best effort): show small ones (<2e4 cells)
                for name, val in list(ns.items()):
                    try:
                        import pandas as pd
                        if isinstance(val, pd.DataFrame):
                            if val.size <= 20000:
                                dfs.append((name, val))
                    except Exception:
                        pass
        except Exception as e:
            out_items.append({'idx': i, 'code': code, 'figs': figs, 'dfs': dfs, 'stdout': stdout.getvalue(), 'error': traceback.format_exc()})
            if stop_on_error:
                break
        else:
            out_items.append({'idx': i, 'code': code, 'figs': figs, 'dfs': dfs, 'stdout': stdout.getvalue(), 'error': None})
    return out_items

code_cells = [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]

if run_btn:
    with st.spinner("正在执行 Notebook 中的代码..."):
        results = execute_cells(code_cells)
    st.success("执行完成")

    # Tabs for organization
    tab_all, tab_figs, tab_tables, tab_cells = st.tabs(["总览", "图表", "表格", "逐单元"])

    # 总览：列出有图/表的单元格摘要
    with tab_all:
        cnt_fig = sum(len(r['figs']) for r in results)
        cnt_df = sum(len(r['dfs']) for r in results)
        st.metric("捕获的图表数量", cnt_fig)
        st.metric("捕获的表格数量", cnt_df)
        st.write("---")
        for r in results:
            if r['error']:
                with st.expander(f"❌ 单元格 #{{r['idx']}} 发生错误"):
                    if show_code: st.code(r['code'], language="python")
                    st.error(r['error'])
                    if r['stdout']:
                        st.text("标准输出：\n" + r['stdout'])
            elif r['figs'] or r['dfs']:
                with st.expander(f"✅ 单元格 #{{r['idx']}}（图:{{len(r['figs'])}} 表:{{len(r['dfs'])}}）"):
                    if show_code: st.code(r['code'], language="python")
                    for fig in r['figs']:
                        st.pyplot(fig)
                    for name, df in r['dfs']:
                        st.caption(f"DataFrame: {{name}}  形状: {{df.shape}}")
                        st.dataframe(df)
                    if r['stdout']:
                        st.text("标准输出：\n" + r['stdout'])

    # 图表：只显示所有 figures
    with tab_figs:
        num = 0
        for r in results:
            for fig in r['figs']:
                num += 1
                st.subheader(f"图表 {{num}}")
                st.pyplot(fig)
        if num == 0:
            st.info("没有捕获到图表。")

    # 表格：只显示 DataFrame（采样显示大表）
    with tab_tables:
        num = 0
        for r in results:
            for name, df in r['dfs']:
                num += 1
                st.subheader(f"表格 {{num}} — {{name}} 形状: {{df.shape}}")
                if df.size > 20000:
                    st.dataframe(df.head(1000))
                    st.caption("表太大，仅显示前 1000 行。")
                else:
                    st.dataframe(df)
        if num == 0:
            st.info("没有捕获到表格。")

    # 逐单元：每个 cell 原样展示
    with tab_cells:
        for r in results:
            header = f"单元格 #{{r['idx']}}"
            if r['error']:
                st.error(header + "（执行出错）")
            else:
                st.markdown("### " + header)
            if show_code:
                st.code(r['code'], language="python")
            if r['stdout']:
                st.text("标准输出：\n" + r['stdout'])
            for fig in r['figs']:
                st.pyplot(fig)
            for name, df in r['dfs']:
                st.caption(f"DataFrame: {{name}}  形状: {{df.shape}}")
                st.dataframe(df)
            if r['error']:
                st.error(r['error'])

else:
    st.info("点击左侧 **▶ 运行 Notebook** 按钮，将执行 `indipro.ipynb` 的代码并捕获真实图表与表格。")
    st.caption("提示：如果 Notebook 需要联网下载数据，请确保 Streamlit Cloud 打开外网访问（默认可以）。")
