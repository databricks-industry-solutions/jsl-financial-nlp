import plotly.graph_objects as go
import networkx as nx
import pandas as pd


def get_nodes_from_graph(graph, pos, node_color):
    """Extracts the nodes from a networkX dataframe in Plotly Scatterplot format"""
    node_x = []
    node_y = []
    texts = []
    hovers = []
    for node in graph.nodes():
        entity = graph.nodes[node]['attr_dict']['entity']
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        texts.append(node)
        hovers.append(entity)

    node_trace = go.Scatter(
        x=node_x, y=node_y, text=texts, hovertext=hovers,
        mode='markers+text',
        hoverinfo='text',
        marker=dict(
            color=node_color,
            size=40,
            line_width=2))

    return node_trace


def get_edges_from_graph(graph, pos, edge_color):
    """Extracts the edges from a networkX dataframe in Plotly Scatterplot format"""
    edge_x = []
    edge_y = []
    hovers = []
    xtext = []
    ytext = []
    for edge in graph.edges():
        relation = graph.edges[edge]['attr_dict']['relation']
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)
        hovers.append(relation)
        xtext.append((x0 + x1) / 2)
        ytext.append((y0 + y1) / 2)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color=edge_color),
        mode='lines')

    labels_trace = go.Scatter(x=xtext, y=ytext, mode='text',
                              textfont={'color': edge_color},
                              marker_size=0.5,
                              text=hovers,
                              textposition='top center',
                              hovertemplate='weight: %{text}<extra></extra>')
    return edge_trace, labels_trace


def show_graph_in_plotly(graph, node_color='white', edge_color='grey'):
    """Shows Plotly graph in Databricks"""
    pos = nx.spring_layout(graph)
    node_trace = get_nodes_from_graph(graph, pos, node_color)
    edge_trace, labels_trace = get_edges_from_graph(graph, pos, edge_color)
    fig = go.Figure(data=[edge_trace, node_trace, labels_trace],
                    layout=go.Layout(
                        title='Company Ecosystem',
                        titlefont_size=16,
                        showlegend=False,
                        width=1600,
                        height=1000,
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    fig.update_traces(marker=dict(size=12,
                                  line=dict(width=2,
                                            color='DarkSlateGrey')),
                      selector=dict(mode='markers'))
    fig.show()


def get_relations_df(results, col='relations'):
    """Shows a Dataframe with the relations extracted by Spark NLP"""
    rel_pairs = []
    for rel in results[0][col]:
        rel_pairs.append((
            rel.result,
            rel.metadata['entity1'],
            rel.metadata['entity1_begin'],
            rel.metadata['entity1_end'],
            rel.metadata['chunk1'],
            rel.metadata['entity2'],
            rel.metadata['entity2_begin'],
            rel.metadata['entity2_end'],
            rel.metadata['chunk2'],
            rel.metadata['confidence']
        ))

    rel_df = pd.DataFrame(rel_pairs,
                          columns=['relation', 'entity1', 'entity1_begin', 'entity1_end', 'chunk1', 'entity2',
                                   'entity2_begin', 'entity2_end', 'chunk2', 'confidence'])

    return rel_df
