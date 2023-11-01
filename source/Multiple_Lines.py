import matplotlib as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go

class MultipleLines:
    def MultipLines(predict, actual, index_test):
        dfpre = pd.DataFrame({'Dự đoán': predict.flatten()})
        dfpre = dfpre.set_index(index_test)
        dfact = pd.DataFrame({'Thực tế': actual.flatten()})
        dfact = dfact.set_index(index_test)

        df_result = pd.concat([dfpre, dfact], axis=1)

        fig = px.line(df_result, x=df_result.index, y=df_result.columns,
                           color_discrete_sequence=["red", "blue"],
                           labels={
                                "Date" : "Ngày",
                                "value" : "Giá trị",
                                "variable" : "Chú thích"
                           }, title="Biểu đồ so sánh kết quả dự đoán và thực tế"
                           )
        fig.update_traces(patch={"line": {"width": 1, "dash": 'dot'}})
        return fig
    def OneLine(data, selected_column_name):
        trace = go.Scatter(x=data.data_old.index ,y=data.data_old[selected_column_name], mode='lines', name='Giá cổ phiếu')
        layout = go.Layout(
            title='Biểu đồ giá cổ phiếu',
            xaxis=dict(title='Ngày'),
            yaxis=dict(title='Giá cổ phiếu'),
            hovermode='closest'
        )
        fig = go.Figure(data=[trace], layout=layout)
        fig.update_layout()  # Kích thước tùy chỉnh 800x400
        return fig