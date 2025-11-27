from PIL import Image
from manim import *
from manim.typing import RGB_Array_Float
from manim.utils.rate_functions import ease_in_out_elastic, ease_in_out_quint
from manim_slides import Slide
import networkx as nx
import random
import numpy as np


class DiapositivasProyecto(Slide):
    max_duration_before_split_reverse = None

    def construct(self):

        Tex.set_default(font_size=48.0, tex_environment="flushleft")

        title = Tex(
            R"Detección de Anomalías en Tráfico de Redes",
            tex_environment="flushleft",
        )
        authors = Tex(
            R"- Mateo Puente, Nicolás Guzmán, Santiago Díaz",
            tex_environment="flushleft",
            font_size=36.0,
        ).to_corner(DOWN + RIGHT)

        self.play(Write(title))
        self.play(Write(authors))

        self.next_slide()

        introTex = Tex("Motivación", font_size=36.0).to_corner(UP + LEFT)
        self.play(ReplacementTransform(title, introTex), Unwrite(authors))
        motivation3 = Tex(
            R"""
            \textbf{Intrusion Detection Systems (IDS)} \\
            - Analizan el tráfico de red y clasifican patrones. \\
            - Los grafos modelan las interacciones entre entidades. \\
            - Nodos: dispositivos, usuarios o IPs. \\
            - Aristas: conexiones o flujo de datos.
            """,
            tex_environment="minipage}{20em}",
            font_size=32.0,
        ).to_edge(LEFT)

        computerIcon = SVGMobject(
            "./assets/computer.svg", fill_color=WHITE
        ).set_background_stroke(width=1)
        userIcon = SVGMobject(
            "./assets/bx-user.svg", fill_color=WHITE
        ).set_background_stroke(width=1)
        serverIcon = SVGMobject(
            "./assets/bx-server.svg", fill_color=WHITE
        ).set_background_stroke(width=1)

        # Now build a graph with these icons as nodes
        randomGraph = nx.fast_gnp_random_graph(15, 0.5, seed=42)

        # Assign the icons to the nodes
        vertex_mobjects = {}

        for i, node in enumerate(randomGraph.nodes()):
            if i % 3 == 0:
                icon = computerIcon.copy().scale(0.1)
            elif i % 3 == 1:
                icon = userIcon.copy().scale(0.1)
            else:
                icon = serverIcon.copy().scale(0.1)
            vertex_mobjects[node] = icon

        graphMobject = (
            Graph.from_networkx(
                randomGraph,
                layout="spring",
                labels=False,
                edge_config={"stroke_color": GRAY_D, "stroke_width": 2},
                vertex_mobjects=vertex_mobjects,
            )
            .scale(1.5)
            .to_edge(RIGHT)
        )

        self.play(Create(graphMobject), duration=1.2, rate_func=ease_in_out_elastic)

        self.play(Write(motivation3), duration=2.4)

        self.next_slide()

        self.play(Unwrite(motivation3))

        # Siguiente diapositiva, el problema

        problemTex = Tex("Planteamiento", font_size=36.0).to_corner(UP + RIGHT)
        self.play(
            ReplacementTransform(
                introTex, problemTex, duration=1.2, rate_func=ease_in_out_quint
            ),
        )

        statement_text = Tex(
            R"""
                \textbf{Problema:} \\
                - Dada una secuencia temporal de grafos que representan el tráfico de red, \\
                identificar cambios estructurales que indiquen la presencia de anomalías o ataques. \\
                - Utilizando técnicas de detección de comunidades y métricas de centralidad.
                """,
            tex_environment="minipage}{20em}",
            font_size=32.0,
        ).to_edge(LEFT)
        randomVertices = random.sample(range(len(randomGraph.nodes())), 3)

        vertexEdges = []
        for v in randomVertices:
            for neighbor in randomGraph.neighbors(v):
                edge = (v, neighbor)
                if edge not in vertexEdges and (neighbor, v) not in vertexEdges:
                    vertexEdges.append(edge)

        self.play(
            *[graphMobject.vertices[v].animate.set_fill(RED_D) for v in randomVertices],
            *[
                graphMobject.vertices[neighbor].animate.set_fill(YELLOW_D)
                for v in randomVertices
                for neighbor in randomGraph.neighbors(v)
                if neighbor not in randomVertices
            ],
            duration=1.2,
        )

        self.play(Write(statement_text), duration=2.4)

        self.next_slide()

        self.play(
            Unwrite(statement_text),
            Uncreate(graphMobject),
        )

        marcoTex = Tex("Marco Teórico / Modularidad", font_size=36.0).to_edge(UP)

        self.play(ReplacementTransform(problemTex, marcoTex), duration=1.2)

        new_graph = nx.fast_gnp_random_graph(15, 0.4, seed=24)

        randomColors = [
            RED,
            GREEN,
            BLUE,
            YELLOW,
            ORANGE,
            PURPLE,
            PINK,
        ]

        randomCommunities = {}
        for node in new_graph.nodes():
            randomCommunities[node] = random.choice(randomColors)
        anotherTex = Tex("Baja Modularidad", font_size=28.0)

        graphMobject2 = Graph.from_networkx(
            new_graph,
            layout="circular",
            labels=False,
            edge_config={"stroke_color": GRAY_D, "stroke_width": 2},
        )

        graphMobject2.scale(1.2)
        self.play(Create(graphMobject2), duration=1.2, rate_func=ease_in_out_elastic)

        self.play(graphMobject2.animate.to_edge(RIGHT).shift(LEFT * 0.5), duration=1.2)

        graphMobject3 = graphMobject2.copy()

        self.add(graphMobject3)

        self.play(graphMobject3.animate.to_edge(LEFT).shift(RIGHT * 0.5), duration=1.2)

        self.play(Write(anotherTex.next_to(graphMobject2, DOWN)), duration=1.2)

        self.play(
            [
                graphMobject2.vertices[node].animate.set_fill(
                    randomCommunities[node], opacity=0.8
                )
                for node in new_graph.nodes()
            ],
            duration=1.2,
        )

        # Now show a high modularity
        communities = nx.community.louvain_communities(new_graph, seed=42)

        newColors = [RED, GREEN, BLUE, YELLOW, ORANGE, PURPLE, PINK][: len(communities)]
        highModularityColors = {}
        for i, community in enumerate(communities):
            for node in community:
                highModularityColors[node] = newColors[i]
        highModularityTex = Tex("Alta Modularidad", font_size=28.0)
        self.next_slide()

        self.play(
            [
                graphMobject3.vertices[node].animate.set_fill(
                    highModularityColors[node], opacity=0.8
                )
                for node in new_graph.nodes()
            ],
            duration=1.2,
        )
        self.play(
            Write(highModularityTex.next_to(graphMobject3, DOWN)),
            duration=1.2,
        )

        self.next_slide()

        self.play(
            Unwrite(anotherTex),
            Unwrite(highModularityTex),
            Uncreate(graphMobject3),
        )

        louvainTitle = Tex("Marco Teórico / Louvain", font_size=36.0).to_edge(UP)

        self.play(TransformMatchingTex(marcoTex, louvainTitle), duration=1.2)

        louvainTex = Tex(
            R"""
            \textbf{Algoritmo de Louvain:} \\
            - Empieza asignándole cada nodo a su propia comunidad. \\
            - Iterativamente agrupa nodos para maximizar la modularidad. \\
            """,
            tex_environment="minipage}{20em}",
            font_size=32.0,
        ).to_edge(LEFT)

        # Animate a step of the Louvain algorithm

        # First, set all nodes to their own community
        for node in new_graph.nodes():
            graphMobject2.vertices[node].set_fill(WHITE, opacity=0.8)

        # Now animate grouping some nodes together

        # Get the colors from the high modularity result

        communityAssignments = {}
        for i, community in enumerate(communities):
            for node in community:
                communityAssignments[node] = i

        # Now for every community, pick a color
        communityColors = {}
        for i in range(len(communities)):
            communityColors[i] = newColors[i]

        # Animate grouping nodes, coloring same community nodes with the same color, at the same time

        self.play(Write(louvainTex))
        # Now finalize with the actual communities found before
        for i in range(len(communities)):
            nodesInCommunity = [
                node for node in new_graph.nodes() if communityAssignments[node] == i
            ]
            self.play(
                *[
                    graphMobject2.vertices[node].animate.set_fill(
                        communityColors[i], opacity=0.8
                    )
                    for node in nodesInCommunity
                ],
                duration=1.2,
            )

        self.next_slide()

        graphMobject2Spring = (
            Graph.from_networkx(
                new_graph,
                layout="spring",
                labels=False,
                edge_config={"stroke_color": GRAY_D, "stroke_width": 2},
            )
            .scale(1.2)
            .to_edge(RIGHT)
        )

        centralidadTex = Tex("Marco Teórico / Centralidad", font_size=36.0).to_edge(UP)

        self.play(ReplacementTransform(louvainTitle, centralidadTex), duration=1.2)
        self.play(
            ReplacementTransform(graphMobject2, graphMobject2Spring),
            duration=1.2,
        )

        cenntralityExplanation = Tex(
            R"""
            \textbf{Centralidad:} \\
            - Mide la importancia de un nodo en la red. \\
            - Los pesos en las aristas representan el volumen de tráfico.
            """,
            tex_environment="minipage}{20em}",
            font_size=32.0,
        ).to_edge(LEFT)

        self.play(
            ReplacementTransform(louvainTex, cenntralityExplanation), duration=1.2
        )

        # Now show that we're doing it but with edge weights
        randomWeights = [random.randint(1, 10) for _ in range(len(new_graph.edges()))]

        # Since the color of the edges is dim, we can overlay a small number on top of each edge
        numberMobjects = [
            Tex(str(weight), font_size=18).set_color(WHITE) for weight in randomWeights
        ]

        for (a, b), edgeMobject in graphMobject2.edges.items():
            edgeIndex = list(new_graph.edges()).index((a, b))
            weightMobject = numberMobjects[edgeIndex]
            weightMobject.move_to(edgeMobject.get_center())

        # Make all weight numbers appear at the same time
        self.play(*[Write(mob) for mob in numberMobjects], duration=1.2)

        # Color the nodes based on centrality with weights
        weightedGraph = new_graph.copy()
        for i, (a, b) in enumerate(weightedGraph.edges()):
            weightedGraph[a][b]["weight"] = randomWeights[i]

        weightedCentrality = nx.betweenness_centrality(weightedGraph, weight="weight")
        maxCentrality = max(weightedCentrality.values())

        # Prepare all node animations to run simultaneously
        node_animations = []
        for node in weightedGraph.nodes():
            intensity = (
                weightedCentrality[node] / maxCentrality if maxCentrality > 0 else 0
            )
            color = interpolate_color(WHITE, RED, intensity)
            node_animations.append(
                graphMobject2Spring.vertices[node].animate.set_fill(color, opacity=0.8)
            )

        # Play all animations simultaneously
        self.play(*node_animations, duration=1.2)

        self.next_slide()

        # Marco Teorico / Grafos Dinamicos
        grafosDinamicosTitle = Tex(
            "Marco Teórico / Grafos Dinámicos", font_size=36.0
        ).to_edge(UP)

        self.play(
            ReplacementTransform(centralidadTex, grafosDinamicosTitle),
            *[Unwrite(mob) for mob in numberMobjects],
            Unwrite(cenntralityExplanation),
            Uncreate(graphMobject2Spring),
            duration=1.2,
        )

        dinamicosExplanation = Tex(
            R"""
            - Grafos que cambian su estructura a lo largo del tiempo. \\
            - Nodos y aristas pueden aparecer, desaparecer o modificarse. \\
            """,
            tex_environment="minipage}{20em}",
            font_size=32.0,
        ).to_edge(LEFT)

        self.play(Write(dinamicosExplanation), duration=2.4)

        # Create a simple dynamic graph visualization
        time_steps = ["t=1", "t=2", "t=3"]

        # Create small graphs for each time step
        graphs_group = VGroup()
        for i, time_label in enumerate(time_steps):
            # Simple graph with increasing complexity
            if i == 0:
                positions = {0: [0, 0.5, 0], 1: [-0.5, -0.5, 0], 2: [0.5, -0.5, 0]}
                edges = [(0, 1), (1, 2)]
            elif i == 1:
                positions = {
                    0: [0, 0.5, 0],
                    1: [-0.5, -0.5, 0],
                    2: [0.5, -0.5, 0],
                    3: [0, -1, 0],
                }
                edges = [(0, 1), (1, 2), (0, 2), (1, 3)]
            else:
                positions = {
                    0: [0, 0.5, 0],
                    1: [-0.5, -0.5, 0],
                    2: [0.5, -0.5, 0],
                    3: [0, -1, 0],
                    4: [-0.8, 0, 0],
                }
                edges = [(0, 1), (1, 2), (0, 2), (1, 3), (2, 3), (0, 4), (4, 1)]

            temp_graph = nx.Graph()
            temp_graph.add_nodes_from(positions.keys())
            temp_graph.add_edges_from(edges)

            small_graph = Graph.from_networkx(
                temp_graph,
                layout=positions,
                labels=False,
                vertex_config={"radius": 0.1, "color": BLUE},
                edge_config={"stroke_color": GRAY, "stroke_width": 2},
            ).scale(0.8)

            time_tex = Tex(time_label, font_size=24).next_to(small_graph, UP, buff=0.1)
            graph_with_label = VGroup(small_graph, time_tex)
            graphs_group.add(graph_with_label)

        graphs_group.arrange(RIGHT, buff=1.0).to_edge(RIGHT)

        for i, graph_group in enumerate(graphs_group):
            self.play(Create(graph_group), duration=0.6)

        self.next_slide()

        # Marco Teorico / Indice de Jaccard
        jaccardTitle = Tex("Marco Teórico / Índice de Jaccard", font_size=36.0).to_edge(
            UP
        )

        self.play(
            ReplacementTransform(grafosDinamicosTitle, jaccardTitle),
            Unwrite(dinamicosExplanation),
            Uncreate(graphs_group),
            duration=1.2,
        )

        jaccardExplanation = Tex(
            R"""
            - Mide la similitud entre dos conjuntos. \\
            - Definido como la intersección sobre la unión de los conjuntos. \\
            """,
            tex_environment="minipage}{20em}",
            font_size=32.0,
        ).to_edge(LEFT)

        self.play(Write(jaccardExplanation), duration=2.4)

        # Create Venn diagram for Jaccard index
        circle_A = Circle(radius=1.0, color=BLUE, fill_opacity=0.3).shift(
            LEFT * 0.5 + RIGHT * 1.5
        )
        circle_B = Circle(radius=1.0, color=RED, fill_opacity=0.3).shift(
            RIGHT * 0.5 + RIGHT * 1.5
        )

        # Labels for the sets
        label_A = Tex("A", font_size=36).next_to(circle_A, LEFT)
        label_B = Tex("B", font_size=36).next_to(circle_B, RIGHT)

        # Group the Venn diagram
        venn_group = (
            VGroup(circle_A, circle_B, label_A, label_B)
            .next_to(jaccardExplanation, RIGHT)
            .shift(RIGHT * 1.5 + UP * 0.5)
        )

        # Jaccard formula
        jaccard_formula = MathTex(
            R"J(A,B) = \frac{|A \cap B|}{|A \cup B|}", font_size=48
        ).next_to(venn_group, DOWN, buff=1.0)

        self.play(
            Create(circle_A),
            Create(circle_B),
            Write(label_A),
            Write(label_B),
            duration=1.2,
        )

        self.play(Write(jaccard_formula), duration=1.2)

        self.next_slide()

        # Nuestro Enfoque
        enfoqueTitle = Tex("Nuestro Enfoque", font_size=36.0).to_edge(UP)

        self.play(
            ReplacementTransform(jaccardTitle, enfoqueTitle),
            Unwrite(jaccardExplanation),
            Uncreate(circle_A),
            Uncreate(circle_B),
            Unwrite(label_A),
            Unwrite(label_B),
            Unwrite(jaccard_formula),
            duration=1.2,
        )

        # Create axes for the dynamic graph
        axes = Axes(
            x_range=[0, 10, 2],
            y_range=[0, 1, 0.2],
            x_length=6,
            y_length=3,
            axis_config={"color": WHITE},
            tips=False,
        )

        # Create labels for the axes
        x_label = Tex("Tiempo", font_size=24).next_to(axes, DOWN).shift(RIGHT * 0.2)
        y_label = (
            Tex("Similitud\\\\de Jaccard", font_size=24)
            .next_to(axes, LEFT)
            .rotate(PI / 2)
        )

        # Create the red dashed threshold line
        threshold_value = 0.6
        threshold_line = DashedLine(
            axes.c2p(0, threshold_value),
            axes.c2p(10, threshold_value),
            color=RED,
            stroke_width=3,
        )
        threshold_label = Tex("Umbral", font_size=20, color=RED).next_to(
            threshold_line, RIGHT
        )

        self.play(Create(axes), Write(x_label), Write(y_label), duration=1.2)
        self.play(Create(threshold_line), Write(threshold_label), duration=0.6)

        # Create ValueTracker for controlling the progress of the graph
        time_tracker = ValueTracker(0)

        # Function to generate y values based on x
        def similarity_function(x):
            if x < 7:
                # Normal behavior - fluctuates around 0.8
                return 0.8 + 0.1 * np.sin(x * 1.5) + 0.03 * np.sin(x * 5)
            else:
                # Anomaly - drops below threshold
                return 0.4 + 0.1 * np.sin(x * 1.5) + 0.03 * np.sin(x * 5)

        # Create the dynamic graph line
        graph_line = VMobject()

        def update_graph_line(mob):
            mob.clear_points()
            current_time = time_tracker.get_value()

            if current_time > 0:
                # Generate points from 0 to current time
                num_points = int(current_time * 15) + 1
                x_vals = np.linspace(0, current_time, num_points)

                points = []
                is_anomaly = False

                for x in x_vals:
                    y = similarity_function(x)
                    y = max(0, min(1, y))  # Clamp between 0 and 1
                    points.append(axes.c2p(x, y))

                    # Check if any point is below threshold (anomaly)
                    if y < threshold_value:
                        is_anomaly = True

                if len(points) > 1:
                    mob.set_points_as_corners(points)

                    # Change color based on anomaly detection
                    if is_anomaly:
                        mob.set_stroke(RED, width=4)  # Red when anomaly detected
                    else:
                        mob.set_stroke(BLUE, width=3)  # Blue when normal

        # Add updater to the graph line
        graph_line.add_updater(update_graph_line)

        # Add the line to scene
        self.add(graph_line.next_to(axes, ORIGIN))

        # Animate the progress from 0 to 10
        self.play(time_tracker.animate.set_value(10), run_time=4, rate_func=linear)

        # Clear updater when animation is done
        graph_line.clear_updaters()

        self.next_slide()

        # Clean up for next slide
        graph_elements = VGroup(
            axes, x_label, y_label, threshold_line, threshold_label, graph_line
        )

        self.play(Uncreate(graph_elements), duration=1.2)

        enfoqueExplanation2 = Tex(
            R"""
            \textbf{Dataset: CIC-IDS2017} \\
            - 2.8 millones de conexiones de red, a lo largo de 5 días. \\
            - Incluye 1 día de tráfico normal y 4 días con varios ataques (DDoS, Brute Force, Port Scanning). \\
            """,
            tex_environment="minipage}{25em}",
            font_size=32.0,
        )

        graphImage = ImageMobject("./assets/Second Dataset/First1H-3n/zoom1.png")

        group = Group(enfoqueExplanation2, graphImage).arrange(LEFT)

        self.play(Write(enfoqueExplanation2), GrowFromCenter(graphImage), duration=2.4)

        self.next_slide()

        self.play(Unwrite(enfoqueExplanation2), FadeOut(graphImage))

        # Resultados
        resultadosTitle = Tex("Resultados Iniciales:", font_size=36.0).to_edge(UP)

        self.play(
            ReplacementTransform(enfoqueTitle, resultadosTitle),
            duration=1.2,
        )

        resultadosContent = Tex(
            R"""
            \textbf{Podemos notar:} \\
            - No se puede distinguir un umbral claro para detectar anomalías. \\
            - La cantidad de ruido en las métricas es alta. \\
            """,
            tex_environment="minipage}{20em}",
            font_size=32.0,
        ).to_edge(LEFT)

        image = (
            ImageMobject("./assets/How does it look.png")
            .scale(0.7)
            .next_to(resultadosContent, RIGHT)
            .shift(RIGHT * 0.5 + DOWN * 0.2)
        )

        self.play(Write(resultadosContent), duration=2.4)

        self.play(GrowFromCenter(image), duration=1.2)

        self.next_slide()

        self.play(FadeOut(image), Unwrite(resultadosContent))

        t2 = Table(
            [
                [
                    "Precisión",
                    "0.523",
                ],
                [
                    "BAS (Balanced Alert Score)",
                    "0.048",
                ],
                [
                    "SNR (Signal to Noise Ratio)",
                    "1.1",
                ],
            ],
            col_labels=[
                Text("Métrica de Evaluación"),
                Text("Valor"),
            ],
            include_outer_lines=True,
            line_config={"stroke_width": 1, "color": GRAY_A},
        ).scale(0.6)

        t2.remove(*t2.get_vertical_lines())

        self.play(Write(t2), duration=1.2)

        self.next_slide()

        resultadosTitle2 = Tex("Comparación con Deep Learning", font_size=36.0).to_edge(
            UP
        )

        self.play(
            ReplacementTransform(resultadosTitle, resultadosTitle2),
            Unwrite(t2),
            duration=1.2,
        )

        attack_labels = [
            "BENIGNO",
            "Bot",
            "Fuerza Bruta",
            "DDoS",
            "DoS",
            "Infiltración",
            "Escaneo de Puertos",
            "Ataque Web",
        ]

        # Confusion matrix values as strings
        data = [
            ["313336", "11357", "15269", "1673", "14482", "88", "5545", "1912"],
            ["1", "314", "0", "0", "0", "0", "0", "0"],
            ["4", "0", "2204", "0", "2", "0", "2", "1"],
            ["166", "4", "0", "20310", "5", "0", "0", "0"],
            ["150", "0", "76", "16", "40148", "0", "0", "36"],
            ["2", "0", "0", "0", "0", "4", "0", "0"],
            ["4", "6", "10", "10", "21", "0", "25375", "3"],
            ["1", "0", "29", "0", "1", "0", "1", "317"],
        ]

        # Add headers (in Spanish)
        col_header = ["Predicción →"] + attack_labels
        table_data = [["Real ↓"] + [""] * len(attack_labels)]  # top-left direction box
        table_data.append(col_header)

        # Now add each row with its class label
        for label, row in zip(attack_labels, data):
            table_data.append([label] + row)

        # Create table
        table = Table(
            table_data,
            include_outer_lines=True,
            line_config={"stroke_width": 1},
        ).scale(0.25)

        # Color coding:
        #   - Diagonal (correct) → GREEN
        #   - False positives → RED (column mismatch)
        #   - False negatives → BLUE  (row mismatch)

        n = len(attack_labels)

        # Diagonal green
        for i in range(n):
            table.get_cell((i + 3, i + 2)).set_color(GREEN)
            # (i+3) because 2 header rows; (i+2) because 1 header col + left box

        # False positives (column-based mistakes)
        for true_idx in range(n):  # rows
            for pred_idx in range(n):  # columns
                if true_idx != pred_idx:
                    cell = table.get_cell((true_idx + 3, pred_idx + 2))
                    # False positive: predicted class = pred_idx
                    #                but true class != pred_idx
                    cell.set_color(RED)

        # False negatives (override with BLUE on row-based missed predictions)
        for true_idx in range(n):
            for pred_idx in range(n):
                if true_idx != pred_idx:
                    cell = table.get_cell((true_idx + 3, pred_idx + 2))
                    # False negative: true class = true_idx
                    #                 but prediction is not true_idx
                    cell.set_color(BLUE)

        table.set_to(ORIGIN)

        otherMetrics = Tex(
            R""" BAS: 0.7753, SNR: \(\approx\) 8.0 """, font_size=32.0
        ).next_to(table, DOWN, buff=0.5)

        self.play(Write(table), duration=2.4)
        self.play(Write(otherMetrics), duration=1.2)

        self.next_slide()

        self.play(Unwrite(table), Unwrite(otherMetrics))

        nuevoDatasetTitle = Tex("Resultados con Nuevo Dataset", font_size=36.0).to_edge(
            UP
        )
        self.play(
            ReplacementTransform(resultadosTitle2, nuevoDatasetTitle),
            duration=1.2,
        )
        NewResultsImage = (
            ImageMobject("./assets/Figure_2.png").scale(0.6).to_edge(RIGHT)
        )

        AnotherIMage = ImageMobject(
            "./assets/Second Dataset/First4H-2Att2Norm-1000n/metrics.png"
        ).scale(0.4)

        group = Group(NewResultsImage, AnotherIMage).arrange(RIGHT).to_edge(RIGHT)

        nuevoDatasetExplanation = (
            Tex(
                R"""
            - Usamos el UNSW-NB15.
            """,
                tex_environment="minipage}{20em}",
                font_size=32.0,
            )
            .next_to(group, DOWN)
            .shift(LEFT * 1.0)
        )

        self.play(GrowFromCenter(group), duration=1.2)

        self.play(Write(nuevoDatasetExplanation), duration=2.4)

        self.next_slide()

        self.play(Unwrite(nuevoDatasetExplanation))
        self.play(FadeOut(group))

        headers = ["Métrica", "Modelo A\n(Anomaly Rules)", "Modelo B\n(Deep Learning)"]

        data = [
            ["Precisión", "0.996", "0.9124"],
            ["Recall", "0.9901", "0.9997"],
            ["BAS", "0.9931", "0.9928"],
            ["SNR", "249", "10.41"],
            ["FPR", "0.0038", "0.0139"],
            ["FNR", "0.0099", "0.00030"],
        ]

        table = Table(
            [headers] + data,
            include_outer_lines=True,
            line_config={"stroke_width": 1},
        ).scale(0.4)

        # --- COLOR CODING ---

        # Green = good performance, Red = worse
        # Modelo A vs Modelo B: compare row by row

        # Helper: color cells by index
        def color_cells(row, colA, colB):
            cellA = table.get_cell((row, colA))
            cellB = table.get_cell((row, colB))

            valA = float(data[row - 2][1])
            valB = float(data[row - 2][2])

            if row in [2, 3, 4]:  # precision, recall, BAS → higher is better
                if valA > valB:
                    cellA.set_fill(GREEN, opacity=0.3)
                    cellB.set_fill(RED, opacity=0.3)
                else:
                    cellA.set_fill(RED, opacity=0.3)
                    cellB.set_fill(GREEN, opacity=0.3)

            else:  # SNR, FPR, FNR → lower is better for last two, higher for SNR
                if row == 5:  # SNR → higher is better
                    if valA > valB:
                        cellA.set_fill(GREEN, opacity=0.3)
                        cellB.set_fill(RED, opacity=0.3)
                    else:
                        cellA.set_fill(RED, opacity=0.3)
                        cellB.set_fill(GREEN, opacity=0.3)
                else:  # FPR / FNR → lower is better
                    if valA < valB:
                        cellA.set_fill(GREEN, opacity=0.3)
                        cellB.set_fill(RED, opacity=0.3)
                    else:
                        cellA.set_fill(RED, opacity=0.3)
                        cellB.set_fill(GREEN, opacity=0.3)

        # Color rows 2–7 (data rows)
        for r in range(2, 8):
            color_cells(r, 2, 3)
        self.play(Write(table), duration=2.4)

        self.next_slide()

        self.play(Unwrite(table))

        title3 = Tex("Comparación de Matrices de Confusión", font_size=36.0).to_edge(UP)
        self.play(
            ReplacementTransform(nuevoDatasetTitle, title3),
            duration=1.2,
        )

        # --- MATRIZ A (Anomaly Rules Model) ---
        # TN = 526, FP = 2
        # FN = 5, TP = 498

        matrixA = Table(
            [["TN\n526", "FP\n2"], ["FN\n5", "TP\n498"]],
            row_labels=[Text("Benigno"), Text("Ataque")],
            col_labels=[Text("Benigno"), Text("Ataque")],
            include_outer_lines=True,
            line_config={"stroke_width": 1},
        ).scale(0.4)

        titleA = Text("Modelo A\n(Reglas + Métricas)", font_size=28).next_to(
            matrixA, UP
        )

        # --- MATRIZ B (Deep Learning Model) ---
        # TN = 437585, FP = 6168
        # FN = 19, TP = 64238

        matrixB = Table(
            [["TN\n437585", "FP\n6168"], ["FN\n19", "TP\n64238"]],
            row_labels=[Text("Benigno"), Text("Ataque")],
            col_labels=[Text("Benigno"), Text("Ataque")],
            include_outer_lines=True,
            line_config={"stroke_width": 1},
        ).scale(0.4)

        titleB = Text("Modelo B\n(Deep Learning)", font_size=28).next_to(matrixB, UP)

        # --- COLOR CODING (green for correct, red for errors) ---

        # Modelo A
        matrixA.get_cell((1, 1)).set_fill(GREEN, opacity=0.3)  # TN
        matrixA.get_cell((1, 2)).set_fill(RED, opacity=0.3)  # FP
        matrixA.get_cell((2, 1)).set_fill(RED, opacity=0.3)  # FN
        matrixA.get_cell((2, 2)).set_fill(GREEN, opacity=0.3)  # TP

        # Modelo B
        matrixB.get_cell((1, 1)).set_fill(GREEN, opacity=0.3)  # TN
        matrixB.get_cell((1, 2)).set_fill(RED, opacity=0.3)  # FP
        matrixB.get_cell((2, 1)).set_fill(RED, opacity=0.3)  # FN
        matrixB.get_cell((2, 2)).set_fill(GREEN, opacity=0.3)  # TP

        # --- POSITIONING ---
        group = VGroup(VGroup(titleA, matrixA), VGroup(titleB, matrixB)).arrange(
            RIGHT, buff=1.5
        )

        self.play(Write(group), duration=2.4)
        self.next_slide()

        self.play(Unwrite(group))

        # Conclusiones
        conclusionesTitle = Tex("Conclusiones", font_size=36.0).to_edge(UP)

        self.play(
            ReplacementTransform(title3, conclusionesTitle),
            Unwrite(resultadosContent),
            duration=1.2,
        )
        t3 = Table(
            [
                ["1-2 minutos", "+10 minutos"],
                ["Alta", "Nula"],
                ["Media (para nosotros)", "Altísima"],
                ["Muy buena", "Excelente"],
            ],
            row_labels=[
                Text("Tiempo de Procesamiento"),
                Text("Interpretabilidad"),
                Text("Complejidad"),
                Text("Resultados"),
            ],
            col_labels=[Text("Nuestro Enfoque"), Text("Deep Learning")],
            top_left_entry=Text("Categorías"),
            include_outer_lines=True,
            line_config={"stroke_width": 1, "color": GRAY_A},
        ).scale(0.5)
        t3.remove(*t3.get_vertical_lines())

        self.play(Write(t3), duration=1.8)

        self.play(t3.animate.next_to(conclusionesTitle, DOWN, buff=0.5))

        self.next_slide()

        self.play(Unwrite(t3))

        # Final slide
        finalTex = Tex(
            R"Gracias por su atención",
            font_size=48.0,
        ).move_to(ORIGIN)

        quip = Tex(
            R"¿Preguntas?",
            font_size=36.0,
        ).next_to(finalTex, DOWN, buff=0.5)

        quip2 = Tex(
            R"991 líneas de Python en estas diapositivas...",
            font_size=24.0,
        ).next_to(quip, DOWN, buff=0.5)

        self.play(
            ReplacementTransform(conclusionesTitle, finalTex),
            duration=1.8,
            rate_func=ease_in_out_elastic,
        )
        self.play(Write(quip), duration=1.2)
        self.play(Write(quip2), duration=1.2)
