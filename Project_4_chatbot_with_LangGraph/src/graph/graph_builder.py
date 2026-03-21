import sys
from src.utils.logger import logging
from src.utils.exception import CustomException
from langgraph.graph import StateGraph, START, END
from src.state.state import State
from src.nodes.basic_chatbot_node import BasicChatbotNode

class GraphBuilder:
    """
    GraphBuilder is responsible for constructing LangGraph workflows.
    
    Attributes:
        llm: The language model instance used by nodes
        graph_builder: Instance of StateGraph to define workflow
    """

    def __init__(self, model):
        try:
            logging.info("Initializing GraphBuilder with provided model")
            self.llm = model
            self.graph_builder = StateGraph(State)

        except Exception as e:
            logging.error("Error while initializing GraphBuilder")
            raise CustomException(e, sys)

    def basic_chatbot_build_graph(self):
        """
        Builds a basic chatbot graph using LangGraph.

        Workflow:
        START --> chatbot node --> END

        Returns:
            Compiled LangGraph object
        """

        try:
            logging.info("Starting to build basic chatbot graph")

            # Initialize chatbot node
            self.basic_chatbot_node = BasicChatbotNode(self.llm)
            logging.info("BasicChatbotNode initialized successfully")

            # Add chatbot node to graph
            self.graph_builder.add_node("chatbot", self.basic_chatbot_node.process)
            logging.info("Chatbot node added to graph")

            # Define graph flow: START -> chatbot -> END
            self.graph_builder.add_edge(START, "chatbot")
            self.graph_builder.add_edge("chatbot", END)
            logging.info("Edges added: START -> chatbot -> END")

            # Compile graph
            graph = self.graph_builder.compile()
            logging.info("Graph compiled successfully")

            return graph

        except Exception as e:
            logging.error("Error while building chatbot graph")
            raise CustomException(e, sys)